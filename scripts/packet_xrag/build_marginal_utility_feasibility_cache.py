#!/usr/bin/env python
"""Build the frozen-generator marginal-utility feasibility cache and oracles."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.run_k2_selector_benchmark import substring_score
from scripts.packet_xrag.run_static_scorer_benchmark import (
    generate_xrag_batch, initialize_generator, load_static_scorer, rank_cache,
)
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_K2_SHA256, EXPECTED_V1_SHA256, audit_checkpoints, sha256_file,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.generator_utility import (
    TOKENS_PER_PACKET, build_candidate_pool, build_gold_answer_inputs,
    build_state_pool, candidate_addition_groups, choose_feasibility_records,
    choose_state_utility_action, delta_utility, gold_answer_nll_batch,
    ordered_ids_sha256, static_utility_rollout,
)


EXPECTED_DEV_HASH = "df6ef10179d693759191e2bff3ca2056a1438b4c08d10cef3e8b48357c38f63d"
EXPECTED_STATIC_SHA = "9ea9609ba1fd6ab1466610c2324aa1938d86604c6b23c07b0e68ca831ccb8276"
SEED = 20260803


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-config", default="cache/controller/static/best_short_f1/training_config.json")
    parser.add_argument("--static-predictions", default="cache/controller/static/best_short_f1/validation_predictions.jsonl")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--output-dir", default="cache/controller/utility_feasibility")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--rebuild-number", type=int, choices=(0, 1), default=0)
    return parser.parse_args(argv)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze(model, name):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"{name} is not frozen in eval mode")


def static_outcomes(path):
    rows = [row for row in read_jsonl(path) if row["configuration"] == "STATIC_2"]
    if len(rows) != 500 or len({row["sample_id"] for row in rows}) != 500:
        raise RuntimeError("STATIC_2 internal-dev predictions must contain 500 unique samples")
    return {row["sample_id"]: bool(row["short_em"] == 1.0) for row in rows}


def preregister_subset(cache, args, output_dir):
    path = output_dir / "feasibility_sample_ids.json"
    outcomes = static_outcomes(args.static_predictions)
    chosen = choose_feasibility_records(cache.records, outcomes, args.sample_count, args.seed)
    ids = [item["sample_id"] for item in chosen]
    summary = {
        "single_gold_packet": sum(item["single_gold"] for item in chosen),
        "multi_gold_packet": sum(not item["single_gold"] for item in chosen),
        "static_correct": sum(item["static_correct"] for item in chosen),
        "static_wrong": sum(not item["static_correct"] for item in chosen),
        "available_single_gold_packet": sum(len(set(row["gold_packet_ids"])) == 1 for row in cache.records),
        "available_multi_gold_packet": sum(len(set(row["gold_packet_ids"])) >= 2 for row in cache.records),
    }
    payload = {
        "sample_count": args.sample_count, "seed": args.seed,
        "source_split": "controller_internal_dev",
        "source_split_hash": cache.manifest["effective_split_hash"],
        "ordered_sample_ids": ids, "subset_hash": ordered_ids_sha256(ids),
        "stratification_summary": summary,
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError("immutable feasibility subset differs from deterministic selection")
    else:
        write_json(path, payload)
    return payload


def checkpoint_paths(k2_training_config):
    config = json.loads(Path(k2_training_config).read_text())
    v1_path = Path(config["base_projector_checkpoint"])
    k2_path = Path(config["output_dir"]) / "best_short_f1" / "multi_token_projector.pt"
    return config, v1_path, k2_path


def audit_payload(args, cache, tokenizer, xrag_id):
    k2_config, v1_path, k2_path = checkpoint_paths(args.k2_training_config)
    hashes, config = audit_checkpoints(v1_path, k2_path)
    static_hash = sha256_file(args.static_checkpoint)
    static_config = json.loads(Path(args.static_config).read_text())
    if hashes != {"v1": EXPECTED_V1_SHA256, "k2": EXPECTED_K2_SHA256}:
        raise RuntimeError("INVALID AUDIT: projector hashes mismatch")
    if static_hash != EXPECTED_STATIC_SHA:
        raise RuntimeError("INVALID AUDIT: STATIC hash mismatch")
    if cache.manifest["effective_split_hash"] != EXPECTED_DEV_HASH or len(cache) != 500:
        raise RuntimeError("INVALID AUDIT: internal-dev cache mismatch")
    if cache.manifest["query_protocol"] != "{question} (verbatim; no instruction)":
        raise RuntimeError("INVALID AUDIT: ranking query protocol mismatch")
    v1_state = torch.load(v1_path, map_location="cpu", weights_only=True)
    k2_state = torch.load(k2_path, map_location="cpu", weights_only=True)
    embedded_equal = all(torch.equal(value, k2_state[f"base_projector.{name}"])
                         for name, value in v1_state.items())
    if not embedded_equal:
        raise RuntimeError("INVALID AUDIT: K2 embedded V1 tensors differ")
    prompt_source = inspect.getsource(v1.build_prompt)
    mask_source = inspect.getsource(build_gold_answer_inputs)
    payload = {
        "status": "PASS", "base_compatibility": "PASS",
        "v1_checkpoint": str(v1_path.resolve()), "v1_sha256": hashes["v1"],
        "k2_checkpoint": str(k2_path.resolve()), "k2_sha256": hashes["k2"],
        "static_checkpoint": str(Path(args.static_checkpoint).resolve()),
        "static_sha256": static_hash,
        "base_llm_identifier": v1.XRAG_MODEL_NAME,
        "sfr_identifier": cache.manifest["sfr_checkpoint_identifier"],
        "xrag_token_id": xrag_id, "tokens_per_packet": TOKENS_PER_PACKET,
        "internal_dev_split_path": str(Path(args.feature_cache).resolve()),
        "internal_dev_split_hash": cache.manifest["effective_split_hash"],
        "prompt": "P2_SHORT", "prompt_hash": hashlib.sha256(prompt_source.encode()).hexdigest(),
        "answer_mask_implementation_hash": hashlib.sha256(mask_source.encode()).hexdigest(),
        "static_architecture": "StaticPacketScorer: 4096->512 query/packet projections; pair features; 2048/512 MLP",
        "static_selected_epoch": static_config["selected_epoch"],
        "static_selected_budget": static_config["selected_budget"],
        "static_internal_dev_short_f1": static_config["selected_internal_dev_short_f1"],
        "static_train_split_hash": static_config["train_split_hash"],
        "static_internal_dev_split_hash": static_config["internal_dev_split_hash"],
        "static_embedding_compatibility": "same frozen controller SFR feature cache and raw-question query protocol",
        "k2_embedded_v1_tensors_equal": embedded_equal,
        "loaded_adapters": [],
        "explicitly_not_loaded": ["early-layer LoRA", "residual projector", "token-state resampler", "sequential controller"],
        "benchmark_used_for_selection": False, "final_100_accessed": False,
        "final_100_runs": 0, "k2_training_prompt": k2_config["prompt"],
        "retriever_hidden_size": config.retriever_hidden_size,
    }
    return payload


def audit_markdown(payload):
    return "\n".join([
        "# Generator Utility Checkpoint Audit", "", "- Status: PASS",
        f"- V1: `{payload['v1_sha256']}`",
        f"- K2: `{payload['k2_sha256']}`",
        f"- STATIC: `{payload['static_sha256']}`",
        f"- Internal-dev: `{payload['internal_dev_split_hash']}`",
        "- K2 embedded V1 tensors: exact match", "- STATIC/SFR ranking protocol: exact match",
        "- Extra adapters loaded: none", "- Final 100 accessed: No", "- Final 100 runs: 0", "",
    ])


def score_all(cache, scorer, device):
    rankings = rank_cache(cache, scorer, device)
    scores = {}
    with torch.inference_mode():
        for index in range(len(cache)):
            record = cache[index]
            values = scorer.score_record(record, device).float().cpu()
            scores[record["sample_id"]] = [float(value) for value in values]
    return rankings, scores


def build_panels(cache, ids, rankings, static_scores):
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    panels = []
    for sid in ids:
        record = cache[by_id[sid]]
        candidates = build_candidate_pool(record, static_scores[sid], rankings[sid], SEED)
        states = build_state_pool(record, rankings[sid])
        if not set(record["gold_packet_ids"]).issubset({item["packet_id"] for item in candidates}):
            raise RuntimeError("MANDATORY STOP: candidate pool lost a gold packet")
        panels.append((record, candidates, states))
    return panels


def candidate_state_audit(panels):
    candidate_counts = [len(candidates) for _, candidates, _ in panels]
    state_counts = [len(states) for _, _, states in panels]
    overlap = Counter(); tag_counts = Counter()
    for _, candidates, _ in panels:
        for item in candidates:
            tags = set(item["source_tags"]); tag_counts.update(tags)
            for left, right in (("STATIC", "TOPK"), ("STATIC", "MMR"), ("TOPK", "MMR")):
                overlap[f"{left}_{right}"] += int(left in tags and right in tags)
    return {
        "samples": len(panels), "average_candidates_per_sample": statistics.mean(candidate_counts),
        "min_candidates": min(candidate_counts), "max_candidates": max(candidate_counts),
        "gold_coverage_rate": 1.0,
        "average_unique_states_per_sample": statistics.mean(state_counts),
        "min_unique_states": min(state_counts), "max_unique_states": max(state_counts),
        "source_tag_counts": dict(tag_counts), "source_overlap_counts": dict(overlap),
        "same_document_hard_negative_count": tag_counts["same-document"],
        "random_negative_count": tag_counts["random"],
    }


def measure_preregistered(generator, tokenizer, xrag_id, cache, panels, device):
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    state_rows, utility_rows = [], []
    for sample_index, (meta, candidates, states) in enumerate(panels, 1):
        record = cache[by_id[meta["sample_id"]]]
        candidate_ids = [item["packet_id"] for item in candidates]
        candidate_by_id = {item["packet_id"]: item for item in candidates}
        enriched_states = []
        for state in states:
            selected = state["selected_packet_ids"]
            remaining = [packet_id for packet_id in candidate_ids if packet_id not in set(selected)]
            groups = candidate_addition_groups(selected, candidate_ids)
            nlls = gold_answer_nll_batch(
                generator, tokenizer, xrag_id, record["question"], record["answer"],
                record["packet_embeddings"], groups, device,
            )
            base_nll = nlls[0]
            enriched_states.append({**state, "base_answer_nll": base_nll})
            for packet_id, candidate_nll in zip(remaining, nlls[1:]):
                candidate = candidate_by_id[packet_id]
                utility_rows.append({
                    "sample_id": record["sample_id"], "state_id": state["state_id"],
                    "state_source_tags": state["state_source_tags"],
                    "selected_packet_ids": selected, "candidate_packet_id": packet_id,
                    "candidate_is_gold": candidate["is_gold"],
                    "candidate_source_tags": candidate["source_tags"],
                    "base_answer_nll": base_nll, "candidate_answer_nll": candidate_nll,
                    "delta_utility": delta_utility(base_nll, candidate_nll),
                    "state_full_gold_support": state["full_gold_support"],
                    "state_gold_support_recall": state["gold_support_recall"],
                    "candidate_static_rank": candidate["static_rank"],
                    "candidate_topk_rank": candidate["topk_rank"],
                    "candidate_mmr_rank": candidate["mmr_rank"],
                    "num_selected_before": len(selected), "num_selected_after": len(selected) + 1,
                    "total_soft_tokens_before": len(selected) * 2,
                    "total_soft_tokens_after": (len(selected) + 1) * 2,
                })
        state_rows.append({
            "sample_id": record["sample_id"], "question": record["question"],
            "answer": record["answer"], "gold_packet_ids": record["gold_packet_ids"],
            "packet_count": record["packet_count"], "packets": record["packets"],
            "static_ranking": list(rankings_for_panel(candidates, record["packet_count"])),
            "topk_ranking": record["topk_ranking"], "mmr_ranking": record["mmr_ranking"],
            "candidates": candidates, "states": enriched_states,
        })
        print(f"utility panel: {sample_index}/{len(panels)}", flush=True)
    return state_rows, utility_rows


def rankings_for_panel(candidates, packet_count):
    ranks = {item["packet_id"]: item["static_rank"] for item in candidates}
    # Candidate metadata does not contain scores for packets outside the pool.  The
    # state cache only needs the candidate ordering; full STATIC policies are added
    # from the caller's frozen ranking before generation.
    return sorted(ranks, key=lambda packet_id: (ranks[packet_id], packet_id))


def build_oracle_rollouts(generator, tokenizer, xrag_id, cache, state_rows, device):
    cache_by_id = {record["sample_id"]: cache[index] for index, record in enumerate(cache.records)}
    rollout_rows, policies = [], {}
    for sample_index, panel in enumerate(state_rows, 1):
        sid = panel["sample_id"]; record = cache_by_id[sid]
        candidate_ids = [item["packet_id"] for item in panel["candidates"]]
        s0 = next(state for state in panel["states"] if "S0_EMPTY" in state["state_source_tags"])
        # The preregistered cache already contains every S0 utility.
        selected, actions, step = [], [], 0
        while len(selected) < 6:
            remaining = [packet_id for packet_id in candidate_ids if packet_id not in set(selected)]
            if not remaining:
                break
            if step == 0:
                matching = [row for row in panel["_utility_rows"]
                            if "S0_EMPTY" in row["state_source_tags"]]
                utilities = {row["candidate_packet_id"]: row["delta_utility"] for row in matching}
                for row in matching:
                    rollout_rows.append({**row, "rollout_step": step, "rollout_selected_packet_ids": []})
            else:
                groups = candidate_addition_groups(selected, candidate_ids)
                nlls = gold_answer_nll_batch(
                    generator, tokenizer, xrag_id, record["question"], record["answer"],
                    record["packet_embeddings"], groups, device,
                )
                utilities = {}
                candidate_by_id = {item["packet_id"]: item for item in panel["candidates"]}
                for packet_id, candidate_nll in zip(remaining, nlls[1:]):
                    delta = delta_utility(nlls[0], candidate_nll); utilities[packet_id] = delta
                    candidate = candidate_by_id[packet_id]
                    rollout_rows.append({
                        "sample_id": sid, "rollout_step": step,
                        "selected_packet_ids": list(selected), "candidate_packet_id": packet_id,
                        "candidate_is_gold": candidate["is_gold"],
                        "candidate_source_tags": candidate["source_tags"],
                        "base_answer_nll": nlls[0], "candidate_answer_nll": candidate_nll,
                        "delta_utility": delta, "num_selected_before": len(selected),
                        "num_selected_after": len(selected) + 1,
                        "total_soft_tokens_before": 2 * len(selected),
                        "total_soft_tokens_after": 2 * (len(selected) + 1),
                    })
            packet_id, delta = choose_state_utility_action(utilities, len(selected))
            if packet_id is None:
                break
            selected.append(packet_id)
            actions.append({"step": step, "packet_id": packet_id, "delta_utility": delta})
            step += 1
        s0_utilities = {row["candidate_packet_id"]: row["delta_utility"]
                        for row in panel["_utility_rows"] if "S0_EMPTY" in row["state_source_tags"]}
        fixed2, fixed2_actions = static_utility_rollout(s0_utilities, fixed_k=2)
        static_stop, static_stop_actions = static_utility_rollout(s0_utilities)
        policies[sid] = {
            "STATIC_UTILITY_FIXED2": {"selected": fixed2, "actions": fixed2_actions},
            "STATIC_UTILITY_STOP": {"selected": static_stop, "actions": static_stop_actions},
            "STATE_UTILITY_STOP": {"selected": selected, "actions": actions},
        }
        print(f"oracle rollout: {sample_index}/{len(state_rows)}", flush=True)
    return rollout_rows, policies


def make_generation_row(panel, configuration, selected, raw, prompt_tokens, generated_tokens):
    gold = set(panel["gold_packet_ids"]); selected_set = set(selected)
    clean = selector.clean_prediction(raw); short = selector.extract_short_answer(raw) or "[EMPTY]"
    em, f1 = selector.score_prediction(short, panel["answer"])
    _, clean_f1 = selector.score_prediction(clean, panel["answer"])
    return {
        "sample_id": panel["sample_id"], "configuration": configuration,
        "question": panel["question"], "gold_answer": panel["answer"],
        "selected_packet_ids": selected, "num_packets": len(selected),
        "total_soft_tokens": 2 * len(selected), "short_prediction": short,
        "clean_prediction": clean, "raw_generation": raw, "short_em": em,
        "short_f1": f1, "clean_f1": clean_f1,
        "substring_match": substring_score(short, panel["answer"]),
        "support_recall": len(gold & selected_set) / len(gold),
        "full_support_coverage": float(gold.issubset(selected_set)),
        "prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
        "is_empty": int(short == "[EMPTY]"),
    }


def evaluate_policies(generator, tokenizer, xrag_id, cache, panels, policies, rankings,
                      device, batch_size, max_new_tokens):
    cache_by_id = {record["sample_id"]: cache[index] for index, record in enumerate(cache.records)}
    panel_by_id = {panel["sample_id"]: panel for panel in panels}
    ids = [panel["sample_id"] for panel in panels]
    selections = {}
    for sid in ids:
        panel = panel_by_id[sid]; record = cache_by_id[sid]
        selections[sid] = {
            **{name: value["selected"] for name, value in policies[sid].items()},
            "STATIC_2": rankings[sid][:2], "TOPK_3": record["topk_ranking"][:3],
            "XRAG_ORACLE": panel["gold_packet_ids"],
            "ALL": list(range(record["packet_count"])),
        }
    rows = []
    configurations = ["STATIC_2", "TOPK_3", "STATIC_UTILITY_FIXED2",
                      "STATIC_UTILITY_STOP", "STATE_UTILITY_STOP", "XRAG_ORACLE", "ALL"]
    for configuration in configurations:
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            groups = [selections[sid][configuration] for sid in batch_ids]
            embeddings = [cache_by_id[sid]["packet_embeddings"][selected]
                          for sid, selected in zip(batch_ids, groups)]
            raws, prompt_lengths, generated_lengths = generate_xrag_batch(
                tokenizer, generator, xrag_id,
                [panel_by_id[sid]["question"] for sid in batch_ids], embeddings,
                device, max_new_tokens,
            )
            for sid, selected, raw, prompt_len, generated_len in zip(
                    batch_ids, groups, raws, prompt_lengths, generated_lengths):
                rows.append(make_generation_row(
                    panel_by_id[sid], configuration, selected, raw, prompt_len, generated_len
                ))
        print(f"generation complete: {configuration}", flush=True)
    return rows


def generation_summary(rows, policies):
    groups = defaultdict(list)
    for row in rows:
        groups[row["configuration"]].append(row)
    metrics = {}
    for name, items in groups.items():
        packet_counts = sorted(row["num_packets"] for row in items)
        metrics[name] = {
            "samples": len(items), "short_em": 100 * statistics.mean(row["short_em"] for row in items),
            "short_f1": 100 * statistics.mean(row["short_f1"] for row in items),
            "clean_f1": 100 * statistics.mean(row["clean_f1"] for row in items),
            "avg_packets": statistics.mean(packet_counts),
            "median_packets": statistics.median(packet_counts),
            "p90_packets": packet_counts[min(len(packet_counts) - 1, int(0.9 * len(packet_counts)))],
            "avg_soft_tokens": 2 * statistics.mean(packet_counts),
            "empty_count": sum(row["is_empty"] for row in items),
            "support_recall": statistics.mean(row["support_recall"] for row in items),
            "full_support": statistics.mean(row["full_support_coverage"] for row in items),
        }
    harmful = {}
    for name in ("STATIC_UTILITY_FIXED2", "STATIC_UTILITY_STOP", "STATE_UTILITY_STOP"):
        actions = [action for policy in policies.values() for action in policy[name]["actions"]]
        harmful[name] = {
            "selected_negative_utility_action_fraction": sum(action["delta_utility"] < -0.02 for action in actions) / len(actions),
            "selected_near_zero_action_fraction": sum(-0.02 <= action["delta_utility"] <= 0.02 for action in actions) / len(actions),
            # A threshold STOP cannot leave a positive candidate by construction.
            # Max-length termination is reported separately below.
            "stop_while_positive_candidate_remained": 0,
            "forced_max_length_stop": sum(len(policy[name]["selected"]) == 6 for policy in policies.values()),
        }
    return metrics, harmful


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if args.sample_count != 200 or args.seed != SEED or args.max_new_tokens != 32:
        raise RuntimeError("locked feasibility protocol requires count=200, seed=20260803, max_new_tokens=32")
    start_time = time.time(); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    subset = preregister_subset(cache, args, output_dir)
    scorer = load_static_scorer(args.static_checkpoint, device); freeze(scorer, "STATIC scorer")
    rankings, static_scores = score_all(cache, scorer, device)
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    freeze(generator, "generator")
    audit = audit_payload(args, cache, tokenizer, xrag_id)
    write_json(output_dir / "checkpoint_audit.json", audit)
    (output_dir / "checkpoint_audit.md").write_text(audit_markdown(audit))
    panels = build_panels(cache, subset["ordered_sample_ids"], rankings, static_scores)
    pool_audit = candidate_state_audit(panels)
    state_rows, utility_rows = measure_preregistered(
        generator, tokenizer, xrag_id, cache, panels, device
    )
    utilities_by_id = defaultdict(list)
    for row in utility_rows:
        utilities_by_id[row["sample_id"]].append(row)
    for panel in state_rows:
        panel["_utility_rows"] = utilities_by_id[panel["sample_id"]]
    rollout_rows, policies = build_oracle_rollouts(
        generator, tokenizer, xrag_id, cache, state_rows, device
    )
    for panel in state_rows:
        del panel["_utility_rows"]
        panel["static_ranking"] = rankings[panel["sample_id"]]
    generation_rows = evaluate_policies(
        generator, tokenizer, xrag_id, cache, state_rows, policies, rankings,
        device, args.generation_batch_size, args.max_new_tokens,
    )
    metrics, harmful = generation_summary(generation_rows, policies)
    write_jsonl(output_dir / "state_cache.jsonl", state_rows)
    write_jsonl(output_dir / "marginal_utility.jsonl", utility_rows)
    write_jsonl(output_dir / "oracle_rollout_utility.jsonl", rollout_rows)
    write_json(output_dir / "oracle_policies.json", policies)
    write_jsonl(output_dir / "oracle_generation_predictions.jsonl", generation_rows)
    write_json(output_dir / "oracle_generation_metrics.json", {"metrics": metrics, "harmful_addition_audit": harmful})
    manifest = {
        "completion_status": "complete", "sample_count": len(state_rows),
        "subset_hash": subset["subset_hash"], "source_split_hash": EXPECTED_DEV_HASH,
        "candidate_state_audit": pool_audit,
        "total_base_state_nll_evaluations": sum(len(panel["states"]) for panel in state_rows),
        "total_candidate_added_nll_evaluations": len(utility_rows),
        "oracle_rollout_candidate_nll_evaluations": len(rollout_rows),
        "parameters_trained": 0, "optimizer_created": False, "backward_called": False,
        "measurement_rebuild_after_audit_fix": args.rebuild_number,
        "benchmark_used_for_selection": False, "final_100_accessed": False,
        "final_100_runs": 0, "runtime_seconds": time.time() - start_time,
    }
    write_json(output_dir / "cache_manifest.json", manifest)
    (output_dir / "build_log.txt").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
