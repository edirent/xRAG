#!/usr/bin/env python
"""Run the single locked Stage-6 full COMPOSITION_DEV evaluation suite."""

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector, make_fused_tokens,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.model import SFR
from src.packet_xrag.composition.fused_xrag_injection import greedy_generate_fused
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


SEED = 20260804


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    return {
        "samples": len(rows),
        "short_f1": 100 * mean(row["short_f1"] for row in rows),
        "short_em": 100 * mean(row["short_em"] for row in rows),
        "clean_f1": 100 * mean(row["clean_f1"] for row in rows),
        "empty": sum(row["is_empty"] for row in rows),
        "input_packets": mean(row["input_packets"] for row in rows),
        "input_packet_soft_tokens": mean(row["input_packet_soft_tokens"] for row in rows),
        "output_fused_tokens": mean(row["output_fused_tokens"] for row in rows),
        "llm_context_evidence_tokens": mean(row["llm_context_evidence_tokens"] for row in rows),
        "support_recall_before_fusion": mean(row["support_recall"] for row in rows),
        "full_support_before_fusion": mean(row["full_support"] for row in rows),
        "mean_fuser_latency_ms": mean(row["fuser_latency_ms"] for row in rows),
        "mean_total_latency_ms": mean(row["total_latency_ms"] for row in rows),
        "peak_vram_gb": max(row["peak_vram_gb"] for row in rows),
    }


def paired_bootstrap(candidate, baseline, samples=10000):
    by_id = {row["sample_id"]: row["short_f1"] for row in baseline}
    differences = [row["short_f1"] - by_id[row["sample_id"]] for row in candidate]
    rng = random.Random(SEED)
    draws = []
    for _ in range(samples):
        draws.append(100 * mean(differences[rng.randrange(len(differences))]
                                for _ in differences))
    draws.sort()
    return {"samples": samples, "delta": 100 * mean(differences),
            "ci95_lower": draws[int(.025 * samples)],
            "ci95_upper": draws[int(.975 * samples)],
            "p_delta_gt_0": sum(value > 0 for value in draws) / samples}


def decode_row(record, selected, generated, tokenizer, configuration, input_tokens,
               output_tokens, context_tokens, fuser_ms, total_ms, peak):
    eos = generated.eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
    length = int(eos[0]) + 1 if len(eos) else len(generated)
    raw = tokenizer.decode(generated[:length], skip_special_tokens=False)
    short = selector.extract_short_answer(raw) or "[EMPTY]"
    clean = selector.clean_prediction(raw)
    em, f1 = selector.score_prediction(short, record["answer"])
    _, clean_f1 = selector.score_prediction(clean, record["answer"])
    gold, chosen = set(record["gold_packet_ids"]), set(selected)
    return {"sample_id": record["sample_id"], "configuration": configuration,
            "selected_packet_ids": list(selected), "short_prediction": short,
            "short_em": em, "short_f1": f1, "clean_f1": clean_f1,
            "is_empty": short == "[EMPTY]", "input_packets": len(selected),
            "input_packet_soft_tokens": input_tokens, "output_fused_tokens": output_tokens,
            "llm_context_evidence_tokens": context_tokens,
            "support_recall": len(gold & chosen) / len(gold),
            "full_support": float(gold.issubset(chosen)),
            "fuser_latency_ms": fuser_ms, "total_latency_ms": total_ms,
            "peak_vram_gb": peak}


def deterministic_orders(selected, sample_id):
    output = {"REVERSE": list(reversed(selected))}
    for variant in range(3):
        value = list(selected)
        random.Random(f"{SEED}:{sample_id}:order:{variant}").shuffle(value)
        output[f"RANDOM_{variant}"] = value
    return output


def stress_groups(records, rankings):
    groups = {"CLEAN": [], "REVERSE": [], "RANDOM_0": [], "RANDOM_1": [],
              "RANDOM_2": [], "DUPLICATE_X2": [], "DISTRACTOR_X4": []}
    for record in records:
        selected = list(rankings[record["sample_id"]][:6])
        groups["CLEAN"].append(selected)
        orders = deterministic_orders(selected, record["sample_id"])
        for name, value in orders.items():
            groups[name].append(value)
        groups["DUPLICATE_X2"].append(selected + [selected[-1], selected[-1]])
        remaining = [index for index in rankings[record["sample_id"]] if index not in selected]
        if not remaining:
            distractors = [selected[-1]] * 4
        else:
            distractors = [remaining[index % len(remaining)] for index in range(4)]
        groups["DISTRACTOR_X4"].append(selected + distractors)
    return groups


@torch.inference_mode()
def evaluate_independent(configuration, records, groups, k2, tokenizer, generator,
                         xrag_id, device, batch_size):
    rows = []
    # Variable packet counts require homogeneous microbatches. Group by count while
    # retaining per-sample identities in the emitted rows.
    buckets = {}
    for record, selected in zip(records, groups):
        buckets.setdefault(len(selected), []).append((record, selected))
    for count, items in sorted(buckets.items()):
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            current_records = [item[0] for item in batch]
            current_groups = [item[1] for item in batch]
            torch.cuda.reset_peak_memory_stats(device); began = time.time()
            projected = torch.stack([k2(record["packet_embeddings"][selected].to(
                device=device, dtype=torch.bfloat16)).reshape(2 * count, 4096)
                for record, selected in batch])
            prompts = tokenizer([v1.build_prompt(record["question"], 2 * count)
                                 for record in current_records], return_tensors="pt",
                                add_special_tokens=False, padding=True).to(device)
            generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
                prompts.attention_mask, xrag_id, projected, max_new_tokens=32)
            torch.cuda.synchronize(device); elapsed = 1000 * (time.time() - began) / len(batch)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            rows.extend(decode_row(record, selected, generated[index], tokenizer,
                configuration, 2 * count, 2 * count, 2 * count, 0.0, elapsed, peak)
                for index, (record, selected) in enumerate(batch))
    return rows


@torch.inference_mode()
def evaluate_fuser(configuration, fuser, records, groups, k2, tokenizer, generator,
                   xrag_id, device, batch_size):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; selected = groups[start:start + batch_size]
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); began = time.time()
        fused = make_fused_tokens(fuser, "C1", batch, selected, k2, device)
        torch.cuda.synchronize(device); fuser_ms = 1000 * (time.time() - began) / len(batch)
        prompts = tokenizer([v1.build_prompt(record["question"], 4) for record in batch],
                            return_tensors="pt", add_special_tokens=False, padding=True).to(device)
        generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
            prompts.attention_mask, xrag_id, fused, max_new_tokens=32)
        torch.cuda.synchronize(device); total_ms = 1000 * (time.time() - began) / len(batch)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.extend(decode_row(record, group, generated[index], tokenizer, configuration,
            2 * len(group), 4, 4, fuser_ms, total_ms, peak)
            for index, (record, group) in enumerate(zip(batch, selected)))
    return rows


@torch.inference_mode()
def evaluate_grouped(records, groups, k2, tokenizer, generator, xrag_id, device, batch_size):
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    embeddings = []
    texts = ["\n".join(record["packets"][index]["encoder_text"] for index in selected)
             for record, selected in zip(records, groups)]
    for start in range(0, len(texts), 8):
        inputs = sfr_tokenizer(texts[start:start + 8], max_length=512, padding=True,
                               truncation=True, return_tensors="pt").to(device)
        values = sfr.get_doc_embedding(input_ids=inputs.input_ids,
                                       attention_mask=inputs.attention_mask)
        embeddings.extend(value.detach().cpu() for value in values)
    del sfr, sfr_tokenizer; gc.collect(); torch.cuda.empty_cache()
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; selected = groups[start:start + batch_size]
        torch.cuda.reset_peak_memory_stats(device); began = time.time()
        values = torch.stack([k2(value.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
                              .reshape(2, 4096) for value in embeddings[start:start + batch_size]])
        prompts = tokenizer([v1.build_prompt(record["question"], 2) for record in batch],
                            return_tensors="pt", add_special_tokens=False, padding=True).to(device)
        generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
            prompts.attention_mask, xrag_id, values, max_new_tokens=32)
        torch.cuda.synchronize(device); total_ms = 1000 * (time.time() - began) / len(batch)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.extend(decode_row(record, group, generated[index], tokenizer,
            "GROUPED_STATIC_6_K2", 2 * len(group), 2, 2, 0.0, total_ms, peak)
            for index, (record, group) in enumerate(zip(batch, selected)))
    return rows


def robustness(metrics, prefix):
    clean = metrics[f"{prefix}_N6"]["short_f1"]
    order_values = [metrics[f"{prefix}_ORDER_{name}"]["short_f1"]
                    for name in ("REVERSE", "RANDOM_0", "RANDOM_1", "RANDOM_2")]
    return {"clean_f1": clean, "order_mean_f1": mean(order_values),
            "order_degradation": clean - mean(order_values),
            "order_f1_variance": mean((value - mean(order_values)) ** 2
                                      for value in order_values),
            "duplicate_degradation": clean - metrics[f"{prefix}_DUPLICATE_X2"]["short_f1"],
            "distractor_degradation": clean - metrics[f"{prefix}_DISTRACTOR_X4"]["short_f1"]}


def reduction(independent, fused, key):
    base = independent[key]
    if base <= 0:
        return 0.0 if fused[key] >= base else float("-inf")
    return (base - fused[key]) / base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root)
    output_dir = root / "dev_full"; output = output_dir / "results.json"
    if output_dir.exists():
        raise RuntimeError("refusing to overwrite locked full DEV evaluation")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    if ledger["usage"]["full_training_runs"] != 2:
        raise RuntimeError("both preregistered full runs must complete before Stage 6")
    if ledger["usage"]["composition_dev_generation"] != 8:
        raise RuntimeError("unexpected DEV evaluation accounting before Stage 6")
    selections = {}
    for run in ("C1_O1", "C1_O1_O3_O4"):
        selection = json.loads((root / f"full/{run}/selection.json").read_text())
        if sha256(selection["best_checkpoint"]) != selection["best_checkpoint_sha256"]:
            raise RuntimeError(f"{run} best checkpoint hash mismatch")
        selections[run] = selection
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    dev_ids = json.loads((root / "splits/composition_dev_ids.json").read_text())["ordered_sample_ids"]
    records = [cache[by_id[sid]] for sid in dev_ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/train_static_scores.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index)) for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device)
    all_rows = []; by_configuration = {}

    def add(rows):
        all_rows.extend(rows); by_configuration[rows[0]["configuration"]] = rows
        print(json.dumps({rows[0]["configuration"]: summarize(rows)}), flush=True)

    # All shared frozen baselines are generated once.
    for breadth in (2, 4, 6, 12):
        groups = [rankings[record["sample_id"]][:breadth] for record in records]
        add(evaluate_independent(f"INDEPENDENT_N{breadth}", records, groups, k2,
                                 tokenizer, generator, xrag_id, device, args.batch_size))
    all_groups = [rankings[record["sample_id"]] for record in records]
    add(evaluate_independent("INDEPENDENT_ALL", records, all_groups, k2, tokenizer,
                             generator, xrag_id, device, args.batch_size))
    stresses = stress_groups(records, rankings)
    for name in ("REVERSE", "RANDOM_0", "RANDOM_1", "RANDOM_2", "DUPLICATE_X2",
                 "DISTRACTOR_X4"):
        add(evaluate_independent(f"INDEPENDENT_{'ORDER_' if name.startswith(('REVERSE', 'RANDOM')) else ''}{name}",
            records, stresses[name], k2, tokenizer, generator, xrag_id, device, args.batch_size))
    add(evaluate_grouped(records, stresses["CLEAN"], k2, tokenizer, generator, xrag_id,
                         device, args.batch_size))

    for run, selection in selections.items():
        payload = torch.load(selection["best_checkpoint"], map_location="cpu", weights_only=True)
        fuser = build_fuser("C1").to(device); fuser.load_state_dict(payload["state_dict"], strict=True)
        fuser.eval(); prefix = f"FUSER_{run}"
        for breadth in (2, 4, 6, 12):
            groups = [rankings[record["sample_id"]][:breadth] for record in records]
            add(evaluate_fuser(f"{prefix}_N{breadth}", fuser, records, groups, k2,
                               tokenizer, generator, xrag_id, device, args.batch_size))
        add(evaluate_fuser(f"{prefix}_ALL", fuser, records, all_groups, k2, tokenizer,
                           generator, xrag_id, device, args.batch_size))
        for name in ("REVERSE", "RANDOM_0", "RANDOM_1", "RANDOM_2", "DUPLICATE_X2",
                     "DISTRACTOR_X4"):
            label = f"ORDER_{name}" if name.startswith(("REVERSE", "RANDOM")) else name
            add(evaluate_fuser(f"{prefix}_{label}", fuser, records, stresses[name], k2,
                               tokenizer, generator, xrag_id, device, args.batch_size))
        del fuser; torch.cuda.empty_cache()

    metrics = {name: summarize(rows) for name, rows in by_configuration.items()}
    independent_robustness = robustness(metrics, "INDEPENDENT")
    model_results = {}
    for run in selections:
        prefix = f"FUSER_{run}"
        robust = robustness(metrics, prefix)
        reductions = {name: reduction(independent_robustness, robust, name)
                      for name in ("order_degradation", "duplicate_degradation",
                                   "distractor_degradation")}
        static2 = metrics["INDEPENDENT_N2"]["short_f1"]
        n2, n6 = metrics[f"{prefix}_N2"]["short_f1"], metrics[f"{prefix}_N6"]["short_f1"]
        bootstrap = paired_bootstrap(by_configuration[f"{prefix}_N6"],
                                     by_configuration["INDEPENDENT_N2"])
        gate_da = n6 - static2 >= 2 and bootstrap["ci95_lower"] > 0 and (
            n6 - metrics["INDEPENDENT_N6"]["short_f1"] >= 4)
        robust_successes = sum(value >= .5 for value in reductions.values())
        empty_reduced = metrics[f"{prefix}_N6"]["empty"] < metrics["INDEPENDENT_N6"]["empty"]
        gate_db = n6 >= static2 - .5 and n2 - n6 <= 1 and (
            robust_successes + int(empty_reduced) >= 2)
        high_options = []
        for breadth in ("N12", "ALL"):
            fused_value = metrics[f"{prefix}_{breadth}"]["short_f1"]
            independent_value = metrics[f"INDEPENDENT_{breadth}"]["short_f1"]
            high_options.append({"breadth": breadth, "f1": fused_value,
                                 "gain_over_independent": fused_value - independent_value,
                                 "near_static2": fused_value >= static2 - 1})
        gate_dc = any(option["gain_over_independent"] >= 10 and option["near_static2"]
                      for option in high_options)
        model_results[run] = {"checkpoint": selections[run]["best_checkpoint"],
            "checkpoint_sha256": selections[run]["best_checkpoint_sha256"],
            "breadth_degradation_n2_to_n6": n6 - n2,
            "independent_composition_penalty_n6": n6 - metrics["INDEPENDENT_N6"]["short_f1"],
            "fixed_bandwidth_gain_n6": n6 - static2, "paired_bootstrap": bootstrap,
            "robustness": robust, "robustness_reductions": reductions,
            "empty_reduced": empty_reduced, "high_breadth": high_options,
            "gates": {"D-A": gate_da, "D-B": gate_db, "D-C": gate_dc},
            "passed": gate_da or gate_db or gate_dc}
    passing = [run for run, result in model_results.items() if result["passed"]]
    # Protocol tie break: highest N6; within 0.5 prefer simpler O1-only objective.
    selected = None
    if passing:
        passing.sort(key=lambda run: metrics[f"FUSER_{run}_N6"]["short_f1"], reverse=True)
        selected = passing[0]
        if "C1_O1" in passing and metrics[f"FUSER_{selected}_N6"]["short_f1"] - metrics[
                "FUSER_C1_O1_N6"]["short_f1"] < .5:
            selected = "C1_O1"
    status = "PASS" if selected else "MANDATORY_STOP_NO_FULL_MODEL_PASSED_DEV"
    result = {"status": status, "split": "COMPOSITION_DEV", "sample_count": len(records),
              "metrics": metrics, "independent_robustness": independent_robustness,
              "models": model_results, "passing_models": passing,
              "selected_candidate": selected, "selection_rule":
              "highest N6; within 0.5 prefer simpler O1-only; then latency",
              "shadow_accessed": False, "benchmark_accessed": False,
              "final_100_accessed": False}
    output_dir.mkdir(parents=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "predictions.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger["usage"]["composition_dev_generation"] += 1
    ledger["stage6_full_dev"] = {"status": status, "selected_candidate": selected,
                                 "passing_models": passing}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "selected_candidate": selected,
                      "models": model_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
