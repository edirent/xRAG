#!/usr/bin/env python
"""Run the one formal benchmark-500 evaluation after all dev selections freeze."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_checkpoint import load_utility_checkpoint
from src.packet_xrag.controller.utility_evaluation import (
    generate_rollout_answers, run_model_rollouts, summarize_generation,
)


MODEL_CONFIGS = {"A": "MODEL_A_UTILITY_STOP", "B": "MODEL_B_STATE_SHIFT_STOP",
                 "C": "MODEL_C_INTERACTION_STOP"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-dir", default="cache/controller/utility_predictor/model_a")
    parser.add_argument("--model-b-dir", default="cache/controller/utility_predictor/model_b")
    parser.add_argument("--model-c-dir", default="cache/controller/utility_predictor/model_c")
    parser.add_argument("--final-model-type", choices=("B", "C"), required=True)
    parser.add_argument("--feature-cache", default="cache/controller/features/benchmark_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/benchmark_static_scores.pt")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--target-stats", default="cache/controller/utility_predictor/utility_target_stats.json")
    parser.add_argument("--output-dir", default="cache/controller/utility_predictor/benchmark")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def fixed_policy(selected, scores=None):
    selected = list(selected); scores = list(scores or [0.0] * len(selected))
    return {"selected_packet_ids": selected,
            "actions": [{"step": step, "packet_id": packet_id,
                         "predicted_delta": float(scores[step]),
                         "selected_packet_ids_before": selected[:step]}
                        for step, packet_id in enumerate(selected)],
            "stop": {"reason": "fixed_policy", "highest_remaining_packet_id": None,
                     "highest_remaining_score": None}, "tau": None}


def write_summary(output_dir, summaries, alias):
    fields = ["configuration", "short_em", "short_f1", "clean_f1", "avg_packets",
              "median_packets", "p90_packets", "avg_soft_tokens", "empty",
              "support_recall", "full_support", "length_distribution"]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name, metrics in summaries.items():
            writer.writerow({"configuration": name, **{key: metrics[key] for key in fields[1:]}})
    lines = ["# Utility Predictor Benchmark", ""]
    for name, metrics in summaries.items():
        lines.append(f"- {name}: Short F1 {metrics['short_f1']:.4f}; avg packets {metrics['avg_packets']:.4f}")
    lines.extend(["", f"- FINAL_UTILITY_STOP aliases {alias}",
                  "- Model/checkpoint/tau selection used internal-dev only.",
                  "- Final 100 accessed: No", "- Final 100 runs: 0", ""])
    (output_dir / "summary.md").write_text("\n".join(lines))
    with (output_dir / "length_distribution.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["configuration", "length", "count"])
        for name, metrics in summaries.items():
            for length, count in metrics["length_distribution"].items():
                writer.writerow([name, length, count])


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); output_dir = Path(args.output_dir)
    if (output_dir / "benchmark_manifest.json").exists():
        raise RuntimeError("refusing to rerun formal benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); torch.cuda.set_device(device)
    cache = ControllerFeatureCache(args.feature_cache)
    if len(cache) != 500 or cache.manifest["effective_split_hash"] != "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052":
        raise RuntimeError("formal benchmark split mismatch")
    static_scores = load_static_score_cache(
        cache, args.static_checkpoint, args.static_score_cache, device
    )
    clip = json.loads(Path(args.target_stats).read_text())["utility_clip_value"]
    tokenizer, generator, xrag_id, checkpoint_hashes = initialize_generator(
        args.k2_training_config, device
    )
    policies_by_config = {}
    selections = {}
    for model_type, model_dir_value in zip(("A", "B", "C"),
                                           (args.model_a_dir, args.model_b_dir, args.model_c_dir)):
        model_dir = Path(model_dir_value)
        frozen = json.loads((model_dir / "frozen_selection.json").read_text())
        selections[model_type] = frozen
        model = load_utility_checkpoint(model_type, frozen["checkpoint"], device)
        policies_by_config[MODEL_CONFIGS[model_type]] = run_model_rollouts(
            model, cache, static_scores, clip, frozen["tau"], device
        )
        del model
    baseline_policies = {name: {} for name in ("TOPK_3", "STATIC_2", "XRAG_ORACLE", "ALL")}
    for index in range(len(cache)):
        record = cache[index]; sid = record["sample_id"]
        static_ranking = sorted(range(record["packet_count"]),
                                key=lambda packet_id: (-float(static_scores[sid][packet_id]), packet_id))
        baseline_policies["TOPK_3"][sid] = fixed_policy(
            record["topk_ranking"][:3], [record["topk_scores"][i] for i in record["topk_ranking"][:3]]
        )
        baseline_policies["STATIC_2"][sid] = fixed_policy(
            static_ranking[:2], [static_scores[sid][i] for i in static_ranking[:2]]
        )
        baseline_policies["XRAG_ORACLE"][sid] = fixed_policy(record["gold_packet_ids"])
        baseline_policies["ALL"][sid] = fixed_policy(range(record["packet_count"]))
    policies_by_config = {**baseline_policies, **policies_by_config}
    all_rows, summaries = [], {}
    order = ["TOPK_3", "STATIC_2", "MODEL_A_UTILITY_STOP", "MODEL_B_STATE_SHIFT_STOP",
             "MODEL_C_INTERACTION_STOP", "XRAG_ORACLE", "ALL"]
    for configuration in order:
        rows = generate_rollout_answers(
            cache, policies_by_config[configuration], tokenizer, generator, xrag_id,
            device, configuration,
        )
        all_rows.extend(rows); summaries[configuration] = summarize_generation(rows)
        print(json.dumps({configuration: summaries[configuration]}, indent=2), flush=True)
    alias = MODEL_CONFIGS[args.final_model_type]
    summaries["FINAL_UTILITY_STOP"] = summaries[alias]
    with (output_dir / "predictions.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(output_dir, summaries, alias)
    manifest = {"completion_status": "complete", "benchmark_runs": 1,
                "benchmark_hash": cache.manifest["effective_split_hash"],
                "frozen_internal_dev_selections": selections,
                "final_model_type": args.final_model_type, "final_configuration_alias": alias,
                "checkpoint_hashes": checkpoint_hashes, "thresholds_changed": False,
                "benchmark_labels_built": False, "final_100_accessed": False,
                "final_100_runs": 0}
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
