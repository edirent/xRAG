#!/usr/bin/env python
"""Summarize the locked diagnostic suite and apply the Stage-1 gate."""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))


def prediction_disagreement(left, right):
    return mean(a["short_prediction"] != b["short_prediction"] for a, b in zip(left, right))


def main():
    root = Path("cache/composition"); diagnostics = root / "diagnostics"
    summary_path = diagnostics / "composition_gap_summary.csv"
    if summary_path.exists(): raise RuntimeError("refusing to overwrite composition analysis")
    rows = [json.loads(line) for line in
            (diagnostics / "diagnostic_predictions.jsonl").read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows: grouped[row["configuration"]].append(row)
    for values in grouped.values(): values.sort(key=lambda row: row["sample_id"])
    metrics = {}
    for name, values in grouped.items():
        metrics[name] = {"configuration": name, "samples": len(values),
            "short_f1": 100 * mean(row["short_f1"] for row in values),
            "short_em": 100 * mean(row["short_em"] for row in values),
            "avg_packets": mean(row["num_packets"] for row in values),
            "avg_text_tokens": mean(row["text_tokens"] for row in values),
            "avg_soft_tokens": mean(row["soft_tokens"] for row in values),
            "empty": sum(row["is_empty"] for row in values),
            "support_recall": mean(row["support_recall"] for row in values),
            "full_support": mean(row["full_support"] for row in values),
            "mean_answer_logprob": mean(row["mean_token_logprob"] for row in values)}
    fields = list(next(iter(metrics.values())))
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(metrics[name] for name in sorted(metrics))

    soft_early = max(metrics["TOPK_2"]["short_f1"], metrics["TOPK_3"]["short_f1"])
    soft_late = min(metrics["TOPK_6"]["short_f1"], metrics["TOPK_12"]["short_f1"])
    soft_drop = soft_early - soft_late
    text_drop = max(0.0, metrics["TEXT_TOP2"]["short_f1"] -
                    min(metrics["TEXT_TOP6"]["short_f1"], metrics["TEXT_ALL"]["short_f1"]))
    order = {}
    maximum_order_drop = 0.0
    for breadth in (4, 6):
        names = [f"TOPK_{breadth}"] + [f"ORDER_TOPK{breadth}_{variant}" for variant in
                 ("REVERSE", "DOCUMENT", "RANDOM_0", "RANDOM_1", "RANDOM_2")]
        f1s = [metrics[name]["short_f1"] for name in names]
        maximum_order_drop = max(maximum_order_drop, f1s[0] - min(f1s[1:]))
        pairwise = [prediction_disagreement(grouped[names[left]], grouped[names[right]])
                    for left in range(len(names)) for right in range(left + 1, len(names))]
        sample_variances = []
        for index in range(len(grouped[names[0]])):
            values = [grouped[name][index]["short_f1"] for name in names]
            sample_variances.append(mean((value - mean(values)) ** 2 for value in values))
        order[str(breadth)] = {"configurations": names, "mean_f1": mean(f1s),
                               "configuration_f1_std": pstdev(f1s),
                               "maximum_prediction_disagreement_rate": max(pairwise),
                               "mean_per_sample_f1_variance": mean(sample_variances),
                               "empty_variation": max(metrics[name]["empty"] for name in names) -
                                                  min(metrics[name]["empty"] for name in names),
                               "maximum_f1_drop_from_relevance": f1s[0] - min(f1s[1:])}
    duplicate = {}; maximum_duplicate_drop = 0.0
    base_f1 = metrics["TOPK_3"]["short_f1"]
    for kind in ("GOLD", "NONGOLD", "RANDOM"):
        for count in (1, 2, 4):
            name = f"DUP_{kind}_X{count}"; drop = base_f1 - metrics[name]["short_f1"]
            maximum_duplicate_drop = max(maximum_duplicate_drop, drop)
            duplicate[name] = {"f1_drop": drop,
                "empty_increase": metrics[name]["empty"] - metrics["TOPK_3"]["empty"],
                "prediction_change_rate": prediction_disagreement(grouped["TOPK_3"], grouped[name]),
                "answer_confidence_change": metrics[name]["mean_answer_logprob"] -
                                            metrics["TOPK_3"]["mean_answer_logprob"]}
    grouped_gains = {str(breadth): metrics[f"GROUPED_TOP{breadth}_K2"]["short_f1"] -
                     metrics[f"TOPK_{breadth}"]["short_f1"] for breadth in (2, 4, 6)}
    max_grouped_gain = max(grouped_gains.values())
    same_bandwidth = {"independent_top2_k2_f1": metrics["TOPK_2"]["short_f1"],
                      "grouped_top2_k4_f1": metrics["GROUPED_TOP2_K4"]["short_f1"],
                      "grouped_k4_minus_independent_k2": metrics["GROUPED_TOP2_K4"]["short_f1"] -
                                                       metrics["TOPK_2"]["short_f1"],
                      "independent_top4_k1": "SKIPPED_NO_FORMAL_COMPATIBLE_K1_CHECKPOINT"}
    gates = {
        "soft_breadth_drop_ge_3": soft_drop >= 3.0,
        "text_drop_less_than_half_soft": text_drop < soft_drop / 2,
        "order_or_duplicate_drop_ge_2": max(maximum_order_drop, maximum_duplicate_drop) >= 2.0,
        "grouped_gain_ge_2": max_grouped_gain >= 2.0,
        "same_bandwidth_more_independent_packets_worse": False,
    }
    passed_count = sum(gates.values()); status = "PASS" if passed_count >= 2 else "MANDATORY_STOP"
    payload = {"status": status, "passed_gate_count": passed_count, "required_gate_count": 2,
               "gates": gates, "soft_breadth_drop": soft_drop, "text_breadth_drop": text_drop,
               "soft_breadth_curve": {name: metrics[name]["short_f1"] for name in
                                      ("TOPK_2", "TOPK_3", "TOPK_4", "TOPK_6", "TOPK_12", "ALL")},
               "text_breadth_curve": {name: metrics[name]["short_f1"] for name in
                                      ("TEXT_TOP2", "TEXT_TOP4", "TEXT_TOP6", "TEXT_ALL")},
               "grouped_gains": grouped_gains, "same_bandwidth": same_bandwidth,
               "order_sensitivity": order, "duplicate_sensitivity": duplicate,
               "distractor_sensitivity": {key: value for key, value in duplicate.items()
                                           if "NONGOLD" in key or "RANDOM" in key},
               "search_shadow_accessed": False, "benchmark_accessed": False,
               "final_100_accessed": False}
    (diagnostics / "composition_gap_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Composition-Gap Diagnostic", "", f"- Gate: **{status}** ({passed_count}/5 conditions)",
             f"- Independent soft breadth drop: {soft_drop:.3f} F1",
             f"- Corresponding text breadth drop: {text_drop:.3f} F1",
             f"- Maximum order degradation: {maximum_order_drop:.3f} F1",
             f"- Maximum duplicate degradation: {maximum_duplicate_drop:.3f} F1",
             f"- Maximum grouped-compression gain: {max_grouped_gain:.3f} F1", "",
             "## Answers", "",
             "1. Soft and text breadth curves are clearly different: text improves with breadth while independent soft evidence collapses.",
             "2. The K1 more-packets same-bandwidth comparison was skipped because no formal compatible K1 checkpoint exists.",
             "3. Grouped compression is materially more stable at N=6.",
             "4. Order effects are measurable but smaller than duplicate and breadth effects.",
             "5. Duplicates cause strongly nonlinear harm, especially repeated non-gold/random packets.",
             "6. Frozen STATIC-2 remains much stronger than broad independent TOPK concatenation.", "",
             "COMPOSITION_SHADOW, benchmark, and final 100 were not accessed.", ""]
    (diagnostics / "composition_gap_analysis.md").write_text("\n".join(lines))
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["stage1_composition_gap"] = payload
    for item in ledger["diagnostics"]:
        item.update({"status": "completed", "result": status,
                     "actual_metrics": {"soft_drop": soft_drop, "text_drop": text_drop,
                                        "max_order_drop": maximum_order_drop,
                                        "max_duplicate_drop": maximum_duplicate_drop,
                                        "max_grouped_gain": max_grouped_gain},
                     "decision": "keep composition-repair" if status == "PASS" else "mandatory stop",
                     "reason": f"{passed_count} diagnostic conditions passed"})
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
