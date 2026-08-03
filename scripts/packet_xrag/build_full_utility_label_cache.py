#!/usr/bin/env python
"""Audit and build sharded train/dev generator marginal-utility labels."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.run_static_scorer_benchmark import (
    initialize_generator, load_static_scorer,
)
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_K2_SHA256, EXPECTED_V1_SHA256, audit_checkpoints, sha256_file,
)
from src.model import SFR
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.generator_utility import (
    TOKENS_PER_PACKET, build_candidate_pool, build_gold_answer_inputs,
    candidate_addition_groups,
    delta_utility, gold_answer_nll_batch,
)
from src.packet_xrag.controller.utility_label_dataset import (
    INDEX_FORMAT, LABEL_FORMAT, build_full_state_pool, utility_target_statistics,
)


EXPECTED = {
    "train": "91a47f422aeebe7f212058330e4c1357651b8c729b62bdec66379df413104840",
    "internal_dev": "df6ef10179d693759191e2bff3ca2056a1438b4c08d10cef3e8b48357c38f63d",
    "benchmark": "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052",
    "quarantine": "b605a3d6679b51ea070d91d602365fca8b87c0c6b654abdb9607ca3e59a6467e",
    "static": "9ea9609ba1fd6ab1466610c2324aa1938d86604c6b23c07b0e68ca831ccb8276",
}
EXPECTED_COUNTS = {"train": 4499, "internal_dev": 500}
SEED = 20260803


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--split", choices=("train", "internal_dev"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="cache/controller/utility_predictor")
    parser.add_argument("--train-cache", default="cache/controller/features/train_features")
    parser.add_argument("--dev-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--benchmark-cache", default="cache/controller/features/benchmark_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-config", default="cache/controller/static/best_short_f1/training_config.json")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--quarantine-file", default="cache/controller/splits/controller_quarantine.json")
    parser.add_argument("--labels-per-shard", type=int, default=25_000)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args(argv)


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def checkpoint_paths(k2_training_config):
    config = json.loads(Path(k2_training_config).read_text())
    v1_path = Path(config["base_projector_checkpoint"])
    k2_path = Path(config["output_dir"]) / "best_short_f1" / "multi_token_projector.pt"
    return config, v1_path, k2_path


@torch.inference_mode()
def run_checkpoint_audit(args):
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    caches = {
        "train": ControllerFeatureCache(args.train_cache),
        "internal_dev": ControllerFeatureCache(args.dev_cache),
        "benchmark": ControllerFeatureCache(args.benchmark_cache),
    }
    for name, cache in caches.items():
        if cache.manifest["effective_split_hash"] != EXPECTED[name]:
            raise RuntimeError(f"MANDATORY STOP: {name} split hash mismatch")
    quarantine_hash = sha256_file(args.quarantine_file)
    quarantine = json.loads(Path(args.quarantine_file).read_text())
    if quarantine_hash != EXPECTED["quarantine"] or [entry["sample_id"] for entry in quarantine["entries"]] != ["5a7b23ca554299042af8f703"]:
        raise RuntimeError("MANDATORY STOP: quarantine mismatch")
    k2_config, v1_path, k2_path = checkpoint_paths(args.k2_training_config)
    hashes, _ = audit_checkpoints(v1_path, k2_path)
    static_hash = sha256_file(args.static_checkpoint)
    if hashes != {"v1": EXPECTED_V1_SHA256, "k2": EXPECTED_K2_SHA256} or static_hash != EXPECTED["static"]:
        raise RuntimeError("MANDATORY STOP: frozen checkpoint hash mismatch")
    v1_state = torch.load(v1_path, map_location="cpu", weights_only=True)
    k2_state = torch.load(k2_path, map_location="cpu", weights_only=True)
    embedded_equal = all(torch.equal(value, k2_state[f"base_projector.{name}"])
                         for name, value in v1_state.items())
    if not embedded_equal:
        raise RuntimeError("MANDATORY STOP: K2 embedded V1 tensors mismatch")
    device = torch.device(args.device); torch.cuda.set_device(device)
    static = load_static_scorer(args.static_checkpoint, device).eval()
    for parameter in static.parameters(): parameter.requires_grad = False
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    generator.eval()
    for parameter in generator.parameters(): parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in generator.parameters()):
        raise RuntimeError("generator is not frozen")
    if any(parameter.requires_grad for parameter in generator.projector.parameters()):
        raise RuntimeError("K2 projector is not frozen")
    del generator
    torch.cuda.empty_cache()
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in sfr.parameters(): parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in sfr.parameters()):
        raise RuntimeError("SFR is not frozen")
    if any(parameter.requires_grad for parameter in static.parameters()):
        raise RuntimeError("STATIC scorer is not frozen")
    del sfr, sfr_tokenizer, static
    torch.cuda.empty_cache()
    feature_identity = {
        cache.manifest["sfr_checkpoint_identifier"] for cache in caches.values()
    }
    query_hashes = {cache.manifest["query_encoding_template_hash"] for cache in caches.values()}
    packet_versions = {cache.manifest["packet_construction_version"] for cache in caches.values()}
    if len(feature_identity) != 1 or len(query_hashes) != 1 or len(packet_versions) != 1:
        raise RuntimeError("MANDATORY STOP: STATIC feature pipeline mismatch")
    static_config = json.loads(Path(args.static_config).read_text())
    payload = {
        "status": "PASS", "compatibility": "PASS",
        "v1_path": str(v1_path.resolve()), "v1_sha256": hashes["v1"],
        "k2_path": str(k2_path.resolve()), "k2_sha256": hashes["k2"],
        "static_path": str(Path(args.static_checkpoint).resolve()), "static_sha256": static_hash,
        "base_llm_identifier": v1.XRAG_MODEL_NAME, "sfr_identifier": v1.SFR_MODEL_NAME,
        "xrag_token_id": xrag_id, "tokens_per_packet": TOKENS_PER_PACKET,
        "prompt": "P2_SHORT",
        "prompt_hash": hashlib.sha256(inspect.getsource(v1.build_prompt).encode()).hexdigest(),
        "answer_mask_implementation_hash": hashlib.sha256(
            inspect.getsource(build_gold_answer_inputs).encode()
        ).hexdigest(),
        "effective_train_hash": caches["train"].manifest["effective_split_hash"],
        "internal_dev_hash": caches["internal_dev"].manifest["effective_split_hash"],
        "benchmark_hash": caches["benchmark"].manifest["effective_split_hash"],
        "quarantine_hash": quarantine_hash,
        "quarantined_sample_ids": ["5a7b23ca554299042af8f703"],
        "k2_embedded_v1_tensors_equal": embedded_equal,
        "static_train_hash": static_config["train_split_hash"],
        "static_dev_hash": static_config["internal_dev_split_hash"],
        "feature_pipeline": {"sfr_identifiers": sorted(feature_identity),
                             "query_template_hashes": sorted(query_hashes),
                             "packet_construction_versions": sorted(packet_versions)},
        "trainable_generator_parameters": 0, "trainable_sfr_parameters": 0,
        "trainable_k2_parameters": 0, "trainable_static_parameters": 0,
        "loaded_adapters": [], "benchmark_labels_built": False,
        "final_100_accessed": False, "final_100_runs": 0,
        "k2_training_prompt": k2_config["prompt"],
    }
    write_json(output_root / "checkpoint_audit.json", payload)
    lines = ["# Utility Predictor Checkpoint Audit", "", "- Status: PASS",
             f"- V1: `{hashes['v1']}`", f"- K2: `{hashes['k2']}`",
             f"- STATIC: `{static_hash}`", f"- Effective train: `{EXPECTED['train']}`",
             f"- Internal dev: `{EXPECTED['internal_dev']}`",
             f"- Benchmark: `{EXPECTED['benchmark']}`", "- Frozen parameter assertions: PASS",
             "- Extra adapters: none", "- Final 100 accessed: No", "- Final 100 runs: 0", ""]
    (output_root / "checkpoint_audit.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2), flush=True)


def cache_for_split(args):
    return ControllerFeatureCache(args.train_cache if args.split == "train" else args.dev_cache)


def score_record(scorer, record, device):
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        scores = scorer.score_record(record, device).float().cpu()
    ranking = sorted(range(len(scores)), key=lambda packet_id: (-float(scores[packet_id]), packet_id))
    return [float(value) for value in scores], ranking


def label_row(record, state, candidate, base_nll, candidate_nll):
    return {
        "sample_id": record["sample_id"], "state_id": state["state_id"],
        "state_source_tags": state["state_source_tags"],
        "selected_packet_ids": state["selected_packet_ids"],
        "candidate_packet_id": candidate["packet_id"],
        "candidate_is_gold": candidate["is_gold"],
        "candidate_source_tags": candidate["source_tags"],
        "base_answer_nll": base_nll, "candidate_answer_nll": candidate_nll,
        "delta_utility": delta_utility(base_nll, candidate_nll),
        "state_gold_recall": state["state_gold_recall"],
        "state_full_gold_support": state["state_full_gold_support"],
        "num_selected": state["num_selected"],
        "static_score": candidate["static_score"],
        "query_cosine": candidate["query_cosine"],
    }


@torch.inference_mode()
def build_worker(args):
    if args.split is None:
        raise ValueError("--split is required for a worker")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid worker index")
    audit = json.loads((Path(args.output_root) / "checkpoint_audit.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("checkpoint audit must pass before label construction")
    cache = cache_for_split(args)
    if len(cache) != EXPECTED_COUNTS[args.split] or cache.manifest["effective_split_hash"] != EXPECTED[args.split]:
        raise RuntimeError("label worker split mismatch")
    device = torch.device(args.device); torch.cuda.set_device(device)
    scorer = load_static_scorer(args.static_checkpoint, device).eval()
    for parameter in scorer.parameters(): parameter.requires_grad = False
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    generator.eval()
    for parameter in generator.parameters(): parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in generator.parameters()):
        raise RuntimeError("generator must remain frozen")
    parts_dir = Path(args.output_root) / "labels" / "_parts" / args.split
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}.jsonl"
    completed = set()
    # A run may be safely repartitioned after interruption.  Every process has
    # a disjoint assignment in the current run and skips all bundles committed
    # by earlier partitions, regardless of their former worker count.
    for existing_part in sorted(parts_dir.glob("worker_*_of_*.jsonl")):
        for line in existing_part.read_text().splitlines():
            if line.strip(): completed.add(json.loads(line)["sample_id"])
    indices = list(range(args.worker_index, len(cache), args.num_workers))
    started = time.time(); newly_completed = 0
    with part_path.open("a") as stream:
        for position, sample_index in enumerate(indices, 1):
            meta = cache.records[sample_index]
            if meta["sample_id"] in completed:
                continue
            record = cache[sample_index]
            static_scores, static_ranking = score_record(scorer, record, device)
            candidates = build_candidate_pool(record, static_scores, static_ranking, SEED)
            candidate_ids = [item["packet_id"] for item in candidates]
            if not set(record["gold_packet_ids"]).issubset(set(candidate_ids)):
                raise RuntimeError("MANDATORY STOP: gold candidate coverage failure")
            s0_groups = candidate_addition_groups([], candidate_ids)
            s0_nlls = gold_answer_nll_batch(
                generator, tokenizer, xrag_id, record["question"], record["answer"],
                record["packet_embeddings"], s0_groups, device,
            )
            s0_utilities = {packet_id: delta_utility(s0_nlls[0], value)
                            for packet_id, value in zip(candidate_ids, s0_nlls[1:])}
            oracle_id = min(candidate_ids, key=lambda packet_id: (-s0_utilities[packet_id], packet_id))
            if s0_utilities[oracle_id] <= 0:
                oracle_id = None
            states, omitted = build_full_state_pool(record, static_ranking, oracle_id)
            candidate_by_id = {item["packet_id"]: item for item in candidates}
            labels = []
            s0_state = next(state for state in states
                            if "S0_EMPTY" in state["state_source_tags"])
            s0_remaining = candidate_ids
            for packet_id, candidate_nll in zip(s0_remaining, s0_nlls[1:]):
                labels.append(label_row(
                    record, s0_state, candidate_by_id[packet_id], s0_nlls[0], candidate_nll
                ))
            combined_groups, state_specs = [], []
            for state in states:
                if "S0_EMPTY" in state["state_source_tags"]:
                    continue
                selected = state["selected_packet_ids"]
                remaining = [packet_id for packet_id in candidate_ids if packet_id not in set(selected)]
                groups = candidate_addition_groups(selected, candidate_ids)
                start = len(combined_groups); combined_groups.extend(groups)
                state_specs.append((state, remaining, start, len(groups)))
            combined_nlls = gold_answer_nll_batch(
                generator, tokenizer, xrag_id, record["question"], record["answer"],
                record["packet_embeddings"], combined_groups, device,
            )
            for state, remaining, start, length in state_specs:
                nlls = combined_nlls[start:start + length]
                for packet_id, candidate_nll in zip(remaining, nlls[1:]):
                    labels.append(label_row(
                        record, state, candidate_by_id[packet_id], nlls[0], candidate_nll
                    ))
            bundle = {
                "sample_index": sample_index, "sample_id": record["sample_id"],
                "packet_count": record["packet_count"], "gold_packet_ids": record["gold_packet_ids"],
                "candidate_packet_ids": candidate_ids, "candidates": candidates,
                "states": states, "omitted_states": omitted, "labels": labels,
            }
            stream.write(json.dumps(bundle, ensure_ascii=False) + "\n"); stream.flush()
            newly_completed += 1
            if newly_completed % args.log_every == 0 or position == len(indices):
                print(json.dumps({"split": args.split, "worker": args.worker_index,
                                  "completed_new": newly_completed, "assigned": len(indices),
                                  "position": position, "runtime_seconds": time.time() - started}), flush=True)


def sha256(path):
    return sha256_file(path)


def finalize_split(args):
    if args.split is None:
        raise ValueError("--split is required for --finalize")
    cache = cache_for_split(args)
    parts_dir = Path(args.output_root) / "labels" / "_parts" / args.split
    requested = [parts_dir / f"worker_{index:02d}_of_{args.num_workers:02d}.jsonl"
                 for index in range(args.num_workers)]
    if not all(path.exists() for path in requested):
        raise RuntimeError("not all current utility label worker parts exist")
    part_paths = sorted(parts_dir.glob("worker_*_of_*.jsonl"))
    bundles = []
    for path in part_paths:
        bundles.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    bundles.sort(key=lambda item: item["sample_index"])
    if len(bundles) != len(cache) or [item["sample_index"] for item in bundles] != list(range(len(cache))):
        raise RuntimeError("utility label worker coverage is incomplete or duplicated")
    split_dir = Path(args.output_root) / "labels" / args.split
    if split_dir.exists(): shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    shards, index_samples = [], []
    stream = None; current_count = 0; shard_index = -1
    try:
        for bundle in bundles:
            labels = bundle.pop("labels")
            if stream is None or current_count + len(labels) > args.labels_per_shard:
                if stream is not None: stream.close()
                shard_index += 1; current_count = 0
                filename = f"shard_{shard_index:04d}.jsonl"
                stream = (split_dir / filename).open("w")
                shards.append({"filename": filename, "label_count": 0})
            start_line = current_count
            for row in labels:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            current_count += len(labels); shards[-1]["label_count"] += len(labels)
            index_samples.append({**bundle, "shard": shards[-1]["filename"],
                                  "start_line": start_line, "label_count": len(labels)})
    finally:
        if stream is not None: stream.close()
    for shard in shards:
        shard["sha256"] = sha256(split_dir / shard["filename"])
    label_count = sum(item["label_count"] for item in index_samples)
    state_count = sum(len(item["states"]) for item in index_samples)
    omitted = Counter(tag for item in index_samples for tag in item["omitted_states"])
    index = {"format": INDEX_FORMAT, "split": args.split, "sample_count": len(index_samples),
             "ordered_sample_ids": [item["sample_id"] for item in index_samples],
             "samples": index_samples}
    write_json(split_dir / "index.json", index)
    audit = json.loads((Path(args.output_root) / "checkpoint_audit.json").read_text())
    manifest = {
        "format": LABEL_FORMAT, "completion_status": "complete", "split": args.split,
        "split_hash": cache.manifest["effective_split_hash"],
        "quarantine_hash": cache.manifest["quarantine_hash"],
        "checkpoint_hashes": {"v1": audit["v1_sha256"], "k2": audit["k2_sha256"],
                              "static": audit["static_sha256"]},
        "prompt_hash": audit["prompt_hash"],
        "answer_mask_hash": audit["answer_mask_implementation_hash"],
        "candidate_pool_config": {"max_candidates": 12, "gold": "all", "static_top": 4,
                                  "topk_top": 4, "mmr_top": 4,
                                  "same_document_max": 2, "random_max": 2, "seed": SEED},
        "state_pool_config": {"max_unique_states": 5,
                              "sources": ["S0_EMPTY", "S_GOLD1", "S_STATIC1", "S_WRONG1",
                                          "S_SUFFICIENT", "S_ORACLE1_positive_only"],
                              "overflow_resolution": "omit distinct S_STATIC1 only"},
        "nll_batching": "S0 full-state batch, then one combined batch for all remaining states",
        "sample_count": len(index_samples), "state_count": state_count,
        "label_count": label_count, "shards": shards,
        "gold_candidate_coverage": 1.0,
        "average_candidates_per_sample": sum(len(item["candidate_packet_ids"]) for item in index_samples) / len(index_samples),
        "average_states_per_sample": state_count / len(index_samples),
        "omitted_state_counts": dict(omitted), "candidate_added_nll_hard_cap": 260_000,
        "benchmark_labels_built": False, "final_100_accessed": False, "final_100_runs": 0,
    }
    labels_root = Path(args.output_root) / "labels"
    write_json(labels_root / f"{args.split}_manifest.json", manifest)
    if args.split == "train":
        deltas = []
        for shard in shards:
            deltas.extend(json.loads(line)["delta_utility"] for line in
                          (split_dir / shard["filename"]).read_text().splitlines())
        write_json(Path(args.output_root) / "utility_target_stats.json",
                   utility_target_statistics(deltas))
    other = "internal_dev" if args.split == "train" else "train"
    other_manifest = labels_root / f"{other}_manifest.json"
    if other_manifest.exists():
        other_payload = json.loads(other_manifest.read_text())
        total_labels = label_count + other_payload["label_count"]
        if total_labels > 260_000:
            raise RuntimeError(f"MANDATORY STOP: label hard cap exceeded: {total_labels}")
        report = ["# Full Generator-Utility Label Cache", "",
                  f"- Train samples/states/labels: {json.loads((labels_root/'train_manifest.json').read_text())['sample_count']} / {json.loads((labels_root/'train_manifest.json').read_text())['state_count']} / {json.loads((labels_root/'train_manifest.json').read_text())['label_count']}",
                  f"- Internal-dev samples/states/labels: {json.loads((labels_root/'internal_dev_manifest.json').read_text())['sample_count']} / {json.loads((labels_root/'internal_dev_manifest.json').read_text())['state_count']} / {json.loads((labels_root/'internal_dev_manifest.json').read_text())['label_count']}",
                  f"- Combined candidate-added NLL: {total_labels}", "- Gold coverage: 100%",
                  "- Benchmark labels: not built", "- Final 100 accessed: No", "- Final 100 runs: 0", ""]
        (labels_root / "build_report.md").write_text("\n".join(report))
    print(json.dumps(manifest, indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.audit_only:
        run_checkpoint_audit(args)
    elif args.finalize:
        finalize_split(args)
    else:
        build_worker(args)


if __name__ == "__main__":
    main()
