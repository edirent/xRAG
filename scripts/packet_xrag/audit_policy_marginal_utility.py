#!/usr/bin/env python
"""Measure true generator utility only for formal benchmark policy decisions."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.generator_utility import gold_answer_nll_batch, utility_label


AUDITED_CONFIGS = ("TOPK_3", "STATIC_2", "MODEL_A_UTILITY_STOP",
                   "MODEL_B_STATE_SHIFT_STOP", "MODEL_C_INTERACTION_STOP")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="cache/controller/utility_predictor/benchmark")
    parser.add_argument("--feature-cache", default="cache/controller/features/benchmark_features")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); root = Path(args.benchmark_dir)
    output = root / "policy_utility_audit.jsonl"
    if output.exists(): raise RuntimeError("refusing to rerun formal policy utility audit")
    predictions = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
    by_key = {(row["configuration"], row["sample_id"]): row for row in predictions}
    cache = ControllerFeatureCache(args.feature_cache)
    device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer, generator, xrag_id, _ = initialize_generator(args.k2_training_config, device)
    records = []
    for index in range(len(cache)):
        record = cache[index]; sid = record["sample_id"]
        for configuration in AUDITED_CONFIGS:
            prediction = by_key[(configuration, sid)]
            decisions = [("selected_action", action["selected_packet_ids_before"],
                          action["packet_id"], action["step"])
                         for action in prediction["actions"]]
            stop = prediction.get("stop")
            if stop and stop.get("highest_remaining_packet_id") is not None:
                decisions.append(("stop_candidate", prediction["selected_packet_ids"],
                                  stop["highest_remaining_packet_id"], None))
            groups = []
            for _, selected, candidate, _ in decisions:
                groups.extend([selected, selected + [candidate]])
            if groups:
                nlls = gold_answer_nll_batch(
                    generator, tokenizer, xrag_id, record["question"], record["answer"],
                    record["packet_embeddings"], groups, device,
                )
            for decision_index, (kind, selected, candidate, step) in enumerate(decisions):
                base, added = nlls[2 * decision_index:2 * decision_index + 2]
                delta = base - added
                records.append({"sample_id": sid, "configuration": configuration,
                                "decision_type": kind, "step": step,
                                "selected_packet_ids": selected,
                                "candidate_packet_id": candidate,
                                "base_answer_nll": base, "candidate_answer_nll": added,
                                "delta_utility": delta, "utility_label": utility_label(delta)})
        if (index + 1) % 25 == 0:
            print(f"policy utility audit: {index + 1}/{len(cache)}", flush=True)
    with output.open("w") as stream:
        for row in records: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {}
    for configuration in AUDITED_CONFIGS:
        selected = [row for row in records if row["configuration"] == configuration and
                    row["decision_type"] == "selected_action"]
        stops = [row for row in records if row["configuration"] == configuration and
                 row["decision_type"] == "stop_candidate"]
        counts = Counter(row["utility_label"] for row in selected)
        summary[configuration] = {
            "selected_actions": len(selected),
            "positive_fraction": counts["positive"] / len(selected),
            "near_zero_fraction": counts["near-zero"] / len(selected),
            "negative_fraction": counts["negative"] / len(selected),
            "harmful_addition_rate": counts["negative"] / len(selected),
            "stop_decisions_audited": len(stops),
            "stop_with_positive_candidate_remaining": sum(row["delta_utility"] > .02 for row in stops),
            "stop_with_only_nonpositive_candidate": sum(row["delta_utility"] <= 0 for row in stops),
            "missed_positive_stop_rate": (sum(row["delta_utility"] > .02 for row in stops) /
                                          len(stops) if stops else None),
            "utility_audit_coverage": 1.0,
        }
    payload = {"status": "PASS", "utility_audit_coverage": 1.0,
               "configurations": summary, "used_for_selection": False,
               "final_100_accessed": False, "final_100_runs": 0}
    (root / "policy_utility_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
