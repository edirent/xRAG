#!/usr/bin/env python
"""Evaluate all Stage-3 architecture probes on the locked DEV diagnostic-150."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.composition_training_common import (
    BRANCHES, build_fuser, load_frozen_generator, load_frozen_k2_projector,
    make_fused_tokens, selected_ids,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.fused_xrag_injection import greedy_generate_fused
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


def summarize(rows):
    return {"samples": len(rows), "short_f1": 100 * mean(row["short_f1"] for row in rows),
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
            "peak_vram_gb": max(row["peak_vram_gb"] for row in rows)}


def prompt_batch(tokenizer, records, device):
    prompts = [v1.build_prompt(record["question"], 4) for record in records]
    return tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                     padding=True).to(device)


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / "probes"
    output = output_dir / "architecture_probe_results.json"
    if output.exists(): raise RuntimeError("refusing to overwrite architecture probe evaluation")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    if ledger["usage"]["diagnostic_probe_runs"] != 5 or ledger["usage"]["composition_dev_generation"] != 1:
        raise RuntimeError("unexpected pre-probe experiment accounting")
    ids = json.loads((root / "diagnostics/diagnostic_ids.json").read_text())["ordered_sample_ids"]
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    records = [cache[by_id[sid]] for sid in ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/train_static_scores.pt", device)
    static_rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
                       key=lambda index: (-float(static_scores[record["sample_id"]][index]), index))
                       for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device)
    prediction_rows = []; baseline_metrics = {}
    # One consolidated missing-baseline suite: independent STATIC4/6.
    for breadth in (4, 6):
        current_rows = []; configuration = f"INDEPENDENT_STATIC_{breadth}"
        for start in range(0, len(records), 16):
            batch = records[start:start + 16]
            selected = [static_rankings[record["sample_id"]][:breadth] for record in batch]
            projected = torch.stack([k2(record["packet_embeddings"][ids_selected].to(
                device=device, dtype=torch.bfloat16)).reshape(2 * len(ids_selected), 4096)
                for record, ids_selected in zip(batch, selected)])
            prompts = tokenizer([v1.build_prompt(record["question"], 2 * breadth) for record in batch],
                                return_tensors="pt", add_special_tokens=False, padding=True).to(device)
            generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
                prompts.attention_mask, xrag_id, projected, max_new_tokens=32)
            for row_index, (record, ids_selected) in enumerate(zip(batch, selected)):
                tokens = generated[row_index]
                eos = tokens.eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
                length = int(eos[0]) + 1 if len(eos) else len(tokens)
                raw = tokenizer.decode(tokens[:length], skip_special_tokens=False)
                short = selector.extract_short_answer(raw) or "[EMPTY]"
                clean = selector.clean_prediction(raw)
                em, f1 = selector.score_prediction(short, record["answer"])
                _, clean_f1 = selector.score_prediction(clean, record["answer"])
                gold, chosen = set(record["gold_packet_ids"]), set(ids_selected)
                current_rows.append({"sample_id": record["sample_id"],
                    "configuration": configuration, "selected_packet_ids": ids_selected,
                    "short_prediction": short, "short_em": em, "short_f1": f1,
                    "clean_f1": clean_f1, "is_empty": short == "[EMPTY]",
                    "support_recall": len(gold & chosen) / len(gold),
                    "full_support": float(gold.issubset(chosen)),
                    "input_packets": len(ids_selected), "input_packet_soft_tokens": 2 * len(ids_selected),
                    "output_fused_tokens": 2 * len(ids_selected),
                    "llm_context_evidence_tokens": 2 * len(ids_selected), "fuser_latency_ms": 0.0,
                    "total_latency_ms": 0.0, "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3})
        prediction_rows.extend(current_rows); baseline_metrics[configuration] = summarize(current_rows)
        print(json.dumps({configuration: baseline_metrics[configuration]}), flush=True)
    diagnostic_rows = [json.loads(line) for line in
                       (root / "diagnostics/diagnostic_predictions.jsonl").read_text().splitlines()]
    diagnostic_summary = defaultdict(list)
    for row in diagnostic_rows: diagnostic_summary[row["configuration"]].append(row)
    static2_f1 = 100 * mean(row["short_f1"] for row in diagnostic_summary["STATIC_2"])
    topk6_f1 = 100 * mean(row["short_f1"] for row in diagnostic_summary["TOPK_6"])
    branch_results = {}
    for branch_id in ("A1", "B1", "C1", "D1"):
        checkpoint_path = output_dir / branch_id / "probe.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        fuser = build_fuser(branch_id).to(device)
        fuser.load_state_dict(payload["state_dict"], strict=True); fuser.eval()
        breadth_metrics = {}; branch_rows = []
        for breadth in (2, 4, 6):
            current_rows = []; configuration = f"{branch_id}_N{breadth}"
            torch.cuda.reset_peak_memory_stats(device)
            for start in range(0, len(records), args.batch_size):
                batch = records[start:start + args.batch_size]
                groups = [selected_ids(record, static_rankings[record["sample_id"]],
                                       branch_id, breadth) for record in batch]
                torch.cuda.synchronize(device); began = time.time()
                fused = make_fused_tokens(fuser, branch_id, batch, groups, k2, device)
                torch.cuda.synchronize(device); fuser_ms = 1000 * (time.time() - began)
                inputs = prompt_batch(tokenizer, batch, device)
                generated_ids = greedy_generate_fused(generator, tokenizer, inputs.input_ids,
                    inputs.attention_mask, xrag_id, fused, max_new_tokens=32)
                torch.cuda.synchronize(device); total_ms = 1000 * (time.time() - began)
                for row_index, (record, selected) in enumerate(zip(batch, groups)):
                    tokens = generated_ids[row_index]
                    eos = tokens.eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
                    length = int(eos[0]) + 1 if len(eos) else len(tokens)
                    raw = tokenizer.decode(tokens[:length], skip_special_tokens=False)
                    short = selector.extract_short_answer(raw) or "[EMPTY]"; clean = selector.clean_prediction(raw)
                    em, f1 = selector.score_prediction(short, record["answer"])
                    _, clean_f1 = selector.score_prediction(clean, record["answer"])
                    gold, chosen = set(record["gold_packet_ids"]), set(selected)
                    current_rows.append({"sample_id": record["sample_id"],
                        "configuration": configuration, "branch": branch_id, "breadth": breadth,
                        "selected_packet_ids": selected, "short_prediction": short,
                        "short_em": em, "short_f1": f1, "clean_f1": clean_f1,
                        "is_empty": short == "[EMPTY]", "input_packets": len(selected),
                        "input_packet_soft_tokens": 2 * len(selected), "output_fused_tokens": 4,
                        "llm_context_evidence_tokens": 4,
                        "support_recall": len(gold & chosen) / len(gold),
                        "full_support": float(gold.issubset(chosen)),
                        "fuser_latency_ms": fuser_ms / len(batch),
                        "total_latency_ms": total_ms / len(batch),
                        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3})
            branch_rows.extend(current_rows); prediction_rows.extend(current_rows)
            breadth_metrics[str(breadth)] = summarize(current_rows)
            print(json.dumps({configuration: breadth_metrics[str(breadth)]}), flush=True)
        n2, n6 = breadth_metrics["2"]["short_f1"], breadth_metrics["6"]["short_f1"]
        independent_n6 = (topk6_f1 if branch_id == "A1" else
                          baseline_metrics["INDEPENDENT_STATIC_6"]["short_f1"])
        gates = {"quality_gain_ge_1": n6 - static2_f1 >= 1.0,
                 "breadth_robust_and_near_static2": n6 >= n2 - .5 and n6 >= static2_f1 - .5,
                 "interference_gain_ge_3": n6 - independent_n6 >= 3.0}
        if branch_id == "C1":
            gates["residual_safety"] = n6 >= static2_f1
        branch_results[branch_id] = {"breadths": breadth_metrics,
            "static2_f1": static2_f1, "independent_n6_f1": independent_n6,
            "n6_minus_static2": n6 - static2_f1,
            "n6_minus_independent_n6": n6 - independent_n6,
            "initialization_static2_exact": branch_id == "C1", "gates": gates,
            "passed": any(gates.values())}
    passed = [branch for branch, result in branch_results.items() if result["passed"]]
    promoted = sorted(passed, key=lambda branch: (
        -branch_results[branch]["breadths"]["6"]["short_f1"],
        0 if branch == "C1" else 1, branch))[:3]
    status = "PASS" if promoted else "MANDATORY_STOP_ALL_ARCHITECTURES_FAILED"
    result = {"status": status, "split": "COMPOSITION_DEV_DIAGNOSTIC_150",
              "static2_f1": static2_f1, "baselines": baseline_metrics,
              "branches": branch_results, "promoted_branches": promoted,
              "probe_runs": 4, "dev_generation_evaluations": 5,
              "shadow_accessed": False, "benchmark_accessed": False,
              "final_100_accessed": False}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "architecture_probe_predictions.jsonl").open("w") as stream:
        for row in prediction_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger["usage"]["diagnostic_probe_runs"] = 9
    ledger["usage"]["composition_dev_generation"] = 6
    ledger["stage3_architecture_probes"] = {"status": status, "promoted_branches": promoted}
    for entry in ledger["branches"]:
        branch = entry["experiment_id"]; report = json.loads(
            (output_dir / branch / "training_report.json").read_text())
        entry.update({"status": "completed", "actual_metrics": branch_results[branch],
                      "decision": "promote" if branch in promoted else "reject",
                      "reason": "passed a Stage-3 gate" if branch_results[branch]["passed"] else
                                "failed all Stage-3 gates",
                      "checkpoint_hash": report["checkpoint_sha256"]})
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
