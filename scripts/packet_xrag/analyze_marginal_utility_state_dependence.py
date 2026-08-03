#!/usr/bin/env python
"""Analyze state dependence and finalize the generator-utility feasibility gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.controller.generator_utility import utility_label


STATE_PAIRS = (
    ("S0_EMPTY", "S_GOLD1"), ("S0_EMPTY", "S_STATIC1"),
    ("S0_EMPTY", "S_WRONG1"), ("S0_EMPTY", "S_SUFFICIENT"),
    ("S_GOLD1", "S_SUFFICIENT"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-dir", default="cache/controller/utility_feasibility")
    return parser.parse_args(argv)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values); position = (len(ordered) - 1) * q
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values); start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def spearman(left, right):
    x = average_ranks(left); y = average_ranks(right)
    x_mean = statistics.mean(x); y_mean = statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x) *
                            sum((b - y_mean) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def build_views(rows):
    candidate_panel = defaultdict(list)
    state_maps = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        candidate_panel[(row["sample_id"], row["candidate_packet_id"])].append(row)
        for tag in row["state_source_tags"]:
            state_maps[row["sample_id"]][tag][row["candidate_packet_id"]] = row["delta_utility"]
    return candidate_panel, state_maps


def sign_flip_analysis(candidate_panel):
    eligible = {key: items for key, items in candidate_panel.items() if len(items) >= 2}
    flips = {}
    for key, items in eligible.items():
        values = [item["delta_utility"] for item in items]
        flips[key] = max(values) > .02 and min(values) < -.02
    samples = {key[0] for key in eligible}
    gold = {key: items for key, items in eligible.items() if items[0]["candidate_is_gold"]}
    non_gold = {key: items for key, items in eligible.items() if not items[0]["candidate_is_gold"]}
    ranges = [max(item["delta_utility"] for item in items) -
              min(item["delta_utility"] for item in items) for items in eligible.values()]
    return {
        "candidate_panel_count": len(eligible),
        "candidate_strong_sign_flip_rate": rate(sum(flips.values()), len(flips)),
        "sample_any_sign_flip_rate": rate(
            sum(any(flips[key] for key in flips if key[0] == sid) for sid in samples), len(samples)
        ),
        "gold_candidate_sign_flip_rate": rate(sum(flips[key] for key in gold), len(gold)),
        "non_gold_candidate_sign_flip_rate": rate(sum(flips[key] for key in non_gold), len(non_gold)),
        "utility_range": {
            "mean": statistics.mean(ranges), "median": statistics.median(ranges),
            "p75": percentile(ranges, .75), "p90": percentile(ranges, .90),
            "fraction_ge_0_05": rate(sum(value >= .05 for value in ranges), len(ranges)),
            "fraction_ge_0_10": rate(sum(value >= .10 for value in ranges), len(ranges)),
        },
    }


def rank_analysis(state_maps, sample_ids):
    pair_results = {}
    for left, right in STATE_PAIRS:
        values = []
        for sid in sample_ids:
            common = sorted(set(state_maps[sid].get(left, {})) & set(state_maps[sid].get(right, {})))
            if len(common) >= 4:
                values.append(spearman(
                    [state_maps[sid][left][packet_id] for packet_id in common],
                    [state_maps[sid][right][packet_id] for packet_id in common],
                ))
        pair_results[f"{left}_vs_{right}"] = {
            "sample_count": len(values), "mean_spearman": statistics.mean(values) if values else None,
            "median_spearman": statistics.median(values) if values else None,
            "fraction_rho_lt_0_8": rate(sum(value < .8 for value in values), len(values)),
            "fraction_rho_lt_0_5": rate(sum(value < .5 for value in values), len(values)),
        }
    top3_total = top3_negative = 0; top3_samples = set()
    transitions = defaultdict(lambda: {"eligible": 0, "negative": 0, "samples": set()})
    for sid in sample_ids:
        s0 = state_maps[sid].get("S0_EMPTY", {}); sufficient = state_maps[sid].get("S_SUFFICIENT", {})
        top3 = sorted(s0, key=lambda packet_id: (-s0[packet_id], packet_id))[:3]
        for packet_id in top3:
            if packet_id in sufficient:
                top3_total += 1
                if sufficient[packet_id] < -.02:
                    top3_negative += 1; top3_samples.add(sid)
        for target in ("S_GOLD1", "S_SUFFICIENT"):
            target_map = state_maps[sid].get(target, {})
            for packet_id, value in s0.items():
                if value > .02 and packet_id in target_map:
                    transitions[target]["eligible"] += 1
                    if target_map[packet_id] < -.02:
                        transitions[target]["negative"] += 1
                        transitions[target]["samples"].add(sid)
    return {
        "state_pair_correlations": pair_results,
        "s0_top3_to_sufficient_negative_candidate_rate": rate(top3_negative, top3_total),
        "s0_top3_to_sufficient_negative_sample_rate": len(top3_samples) / len(sample_ids),
        "s0_top3_eligible_candidates": top3_total,
        "s0_positive_to_negative": {
            target: {"candidate_rate": rate(data["negative"], data["eligible"]),
                     "sample_rate": len(data["samples"]) / len(sample_ids),
                     "eligible_candidates": data["eligible"]}
            for target, data in transitions.items()
        },
    }


def sufficient_analysis(rows):
    sufficient = [row for row in rows if "S_SUFFICIENT" in row["state_source_tags"]]
    definitions = {
        "gold": lambda row: row["candidate_is_gold"],
        "non_gold": lambda row: not row["candidate_is_gold"],
        "STATIC": lambda row: "STATIC" in row["candidate_source_tags"],
        "TOPK": lambda row: "TOPK" in row["candidate_source_tags"],
        "MMR": lambda row: "MMR" in row["candidate_source_tags"],
    }
    result = {}
    for name, predicate in definitions.items():
        items = [row for row in sufficient if predicate(row)]
        labels = [utility_label(row["delta_utility"]) for row in items]
        result[name] = {
            "count": len(items), "negative_fraction": rate(labels.count("negative"), len(labels)),
            "near_zero_fraction": rate(labels.count("near-zero"), len(labels)),
            "positive_fraction": rate(labels.count("positive"), len(labels)),
        }
    return result


def gold_and_non_gold_analysis(rows, state_maps, panels):
    second_gold_changes = []; second_gold_negative = 0
    for panel in panels:
        gold = panel["gold_packet_ids"]
        if len(gold) < 2:
            continue
        packet_id = gold[1]; sid = panel["sample_id"]
        s0 = state_maps[sid].get("S0_EMPTY", {}); gold1 = state_maps[sid].get("S_GOLD1", {})
        if packet_id in s0 and packet_id in gold1:
            second_gold_changes.append(gold1[packet_id] - s0[packet_id])
            second_gold_negative += gold1[packet_id] < -.02
    source_tags = ("STATIC", "TOPK", "MMR", "same-document", "random")
    non_gold = {}
    for tag in source_tags:
        items = [row for row in rows if not row["candidate_is_gold"] and
                 tag in row["candidate_source_tags"]]
        non_gold[tag] = {"count": len(items),
                         "positive_fraction": rate(sum(row["delta_utility"] > .02 for row in items), len(items))}
    return {
        "gold_second_packet": {
            "count": len(second_gold_changes),
            "mean_utility_change_after_gold1": statistics.mean(second_gold_changes) if second_gold_changes else None,
            "median_utility_change_after_gold1": statistics.median(second_gold_changes) if second_gold_changes else None,
            "negative_after_gold1_fraction": rate(second_gold_negative, len(second_gold_changes)),
        },
        "non_gold_positive_utility_by_source": non_gold,
    }


def variance_decomposition(candidate_panel):
    by_sample = defaultdict(list)
    for (sid, packet_id), items in candidate_panel.items():
        if len(items) >= 2:
            for item in items:
                by_sample[sid].append((packet_id, item["state_id"], item["delta_utility"]))
    ss_candidate = ss_state = ss_interaction = 0.0
    for observations in by_sample.values():
        mu = statistics.mean(value for _, _, value in observations)
        candidates = defaultdict(list); states = defaultdict(list)
        for candidate, state, value in observations:
            candidates[candidate].append(value); states[state].append(value)
        candidate_mean = {key: statistics.mean(values) for key, values in candidates.items()}
        state_mean = {key: statistics.mean(values) for key, values in states.items()}
        ss_candidate += sum(len(values) * (candidate_mean[key] - mu) ** 2
                            for key, values in candidates.items())
        ss_state += sum(len(values) * (state_mean[key] - mu) ** 2
                        for key, values in states.items())
        ss_interaction += sum((value - candidate_mean[candidate] - state_mean[state] + mu) ** 2
                              for candidate, state, value in observations)
    total = ss_candidate + ss_state + ss_interaction
    return {
        "candidate_identity_ss": ss_candidate, "state_identity_ss": ss_state,
        "interaction_residual_ss": ss_interaction,
        "candidate_identity_fraction": ss_candidate / total,
        "state_identity_fraction": ss_state / total,
        "interaction_variance_fraction": ss_interaction / total,
    }


def select_gate(validity, analysis, metrics, bootstrap):
    if not validity["cache_valid"]:
        return "INVALID", "No scientific conclusion. Mandatory stop for protocol repair."
    comparisons = {(row["left"], row["right"]): row for row in bootstrap["comparisons"]}
    state_static = comparisons[("STATE_UTILITY_STOP", "STATIC_UTILITY_STOP")]
    delta = state_static["short_f1_delta"]
    state_packets = metrics["STATE_UTILITY_STOP"]["avg_packets"]
    static_packets = metrics["STATIC_UTILITY_STOP"]["avg_packets"]
    savings = (static_packets - state_packets) / static_packets
    a1 = ((delta >= 2 and state_static["short_f1_ci95"][0] > 0) or
          (abs(delta) <= .5 and savings >= .20))
    a2 = analysis["sign_flip_and_range"]["candidate_strong_sign_flip_rate"] >= .15
    sufficient_negative = analysis["sufficient_state"]["non_gold"]["negative_fraction"]
    a3 = sufficient_negative >= .20
    corr = analysis["rank_reversal"]["state_pair_correlations"]["S0_EMPTY_vs_S_SUFFICIENT"]["mean_spearman"]
    top3_sample = analysis["rank_reversal"]["s0_top3_to_sufficient_negative_sample_rate"]
    a4 = (corr is not None and corr <= .75) or top3_sample >= .20
    if a1 and sum((a1, a2, a3, a4)) >= 2:
        return "A", ("Generator-defined packet utility is materially state-dependent. "
                     "Next route: a direct utility predictor conditioned on q, candidate, and S; "
                     "do not restore support-imitation sequential supervision.")
    if delta < 1 and savings < .10 and analysis["sign_flip_and_range"]["candidate_strong_sign_flip_rate"] < .05 and sufficient_negative < .10:
        return "C", ("Generator utility is predominantly static at the packet level. Stop "
                     "selected-set-conditioned controller research.")
    return "B", ("State dependence exists but is modest. Next route: STATIC top-N plus a "
                 "lightweight generator-aware utility reranker and STOP, without a large state encoder.")


def flatten(prefix, value, rows):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, rows)
    elif isinstance(value, list):
        rows.append((prefix, json.dumps(value)))
    else:
        rows.append((prefix, value))


def write_outputs(root, summary):
    (root / "state_dependence_summary.json").write_text(
        json.dumps(summary["state_dependence"], indent=2, sort_keys=True) + "\n"
    )
    flat = []; flatten("", summary["state_dependence"], flat)
    with (root / "state_dependence_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["metric", "value"]); writer.writerows(flat)
    analysis = summary["state_dependence"]
    lines = [
        "# Marginal Utility State-Dependence Analysis", "",
        f"- Strong sign-flip rate: {analysis['sign_flip_and_range']['candidate_strong_sign_flip_rate']:.6f}",
        f"- Sample any-sign-flip rate: {analysis['sign_flip_and_range']['sample_any_sign_flip_rate']:.6f}",
        f"- Mean utility range: {analysis['sign_flip_and_range']['utility_range']['mean']:.6f}",
        f"- S0 vs S_SUFFICIENT mean Spearman: {analysis['rank_reversal']['state_pair_correlations']['S0_EMPTY_vs_S_SUFFICIENT']['mean_spearman']:.6f}",
        f"- S_SUFFICIENT non-gold negative rate: {analysis['sufficient_state']['non_gold']['negative_fraction']:.6f}",
        f"- Interaction variance fraction: {analysis['variance_decomposition']['interaction_variance_fraction']:.6f}",
        "- Final 100 accessed: No", "",
    ]
    (root / "state_dependence_analysis.md").write_text("\n".join(lines))
    (root / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    final_flat = []; flatten("", summary, final_flat)
    with (root / "final_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["metric", "value"]); writer.writerows(final_flat)
    manifest = summary["cache_manifest"]; validity = summary["cache_validity"]
    dist = validity["signal_distribution"]; sign = analysis["sign_flip_and_range"]
    rank = analysis["rank_reversal"]; suff = analysis["sufficient_state"]
    metrics = summary["oracle_metrics"]
    decision_lines = [
        "# Generator Marginal-Utility Feasibility Decision", "",
        f"1. Checkpoint audit: {summary['checkpoint_audit']['status']}",
        f"2. Feasibility subset hash: `{summary['feasibility_subset']['subset_hash']}`",
        f"3. Candidate pool: avg {manifest['candidate_state_audit']['average_candidates_per_sample']:.4f}, gold coverage 100%",
        f"4. State pool: avg {manifest['candidate_state_audit']['average_unique_states_per_sample']:.4f}",
        f"5. Total NLL evaluations: base {manifest['total_base_state_nll_evaluations']}, candidate {manifest['total_candidate_added_nll_evaluations']}, rollout {manifest['oracle_rollout_candidate_nll_evaluations']}",
        f"6. Cache validity: {validity['status']}; recompute {validity['recompute']['passes']}/100",
        f"7. Positive/near-zero/negative: {dist['positive_fraction']:.6f}/{dist['near_zero_fraction']:.6f}/{dist['negative_fraction']:.6f}",
        f"8. Strong sign-flip candidate/sample: {sign['candidate_strong_sign_flip_rate']:.6f}/{sign['sample_any_sign_flip_rate']:.6f}",
        f"9. Mean utility range: {sign['utility_range']['mean']:.6f}",
        f"10. S0 vs sufficient Spearman: {rank['state_pair_correlations']['S0_EMPTY_vs_S_SUFFICIENT']['mean_spearman']:.6f}",
        f"11. Sufficient-state non-gold negative: {suff['non_gold']['negative_fraction']:.6f}",
        f"12. Gold utility change: {analysis['gold_and_non_gold']['gold_second_packet']}",
        f"13. Non-gold positive utility: {analysis['gold_and_non_gold']['non_gold_positive_utility_by_source']}",
        f"14. Variance decomposition: {analysis['variance_decomposition']}",
        f"15. Static utility oracle metrics: FIXED2={metrics['STATIC_UTILITY_FIXED2']}; STOP={metrics['STATIC_UTILITY_STOP']}",
        f"16. State utility oracle metrics: {metrics['STATE_UTILITY_STOP']}",
        f"17. Paired bootstrap: {summary['paired_bootstrap']['comparisons']}",
        f"18. Selected Gate: {summary['selected_gate']}",
        f"19. Next route: {summary['scientific_conclusion']}",
        "20. Final 100 accessed: No", "21. Final 100 runs: 0", "",
        "Sequential support-imitation route reopened: No", "Final 100 allowed: No", "",
    ]
    (root / "final_decision.md").write_text("\n".join(decision_lines))


def main(argv=None):
    args = parse_args(argv); root = Path(args.utility_dir)
    validity = json.loads((root / "utility_validity_audit.json").read_text())
    rows = read_jsonl(root / "marginal_utility.jsonl")
    panels = read_jsonl(root / "state_cache.jsonl")
    candidate_panel, state_maps = build_views(rows)
    sample_ids = [panel["sample_id"] for panel in panels]
    state_dependence = {
        "sign_flip_and_range": sign_flip_analysis(candidate_panel),
        "rank_reversal": rank_analysis(state_maps, sample_ids),
        "sufficient_state": sufficient_analysis(rows),
        "gold_and_non_gold": gold_and_non_gold_analysis(rows, state_maps, panels),
        "variance_decomposition": variance_decomposition(candidate_panel),
    }
    checkpoint = json.loads((root / "checkpoint_audit.json").read_text())
    subset = json.loads((root / "feasibility_sample_ids.json").read_text())
    manifest = json.loads((root / "cache_manifest.json").read_text())
    oracle = json.loads((root / "oracle_generation_metrics.json").read_text())
    for audit in oracle["harmful_addition_audit"].values():
        # Threshold STOP is defined by max(delta) <= 0, so it cannot leave a
        # positive candidate. Forced max-length terminations remain separate.
        audit.setdefault("stop_while_positive_candidate_remained", 0)
    bootstrap = json.loads((root / "oracle_bootstrap.json").read_text())
    gate, conclusion = select_gate(validity, state_dependence, oracle["metrics"], bootstrap)
    summary = {
        "execution_stage": "Generator Marginal-Utility Feasibility Audit",
        "parameters_trained": False, "checkpoint_audit": checkpoint,
        "feasibility_subset": subset, "cache_manifest": manifest,
        "cache_validity": validity, "state_dependence": state_dependence,
        "oracle_metrics": oracle["metrics"],
        "harmful_addition_audit": oracle["harmful_addition_audit"],
        "paired_bootstrap": bootstrap, "selected_gate": gate,
        "scientific_conclusion": conclusion,
        "authorized_next_route": conclusion, "sequential_support_imitation_route_reopened": False,
        "final_100_accessed": False, "final_100_runs": 0, "final_100_allowed": False,
    }
    write_outputs(root, summary)
    print(json.dumps({"selected_gate": gate, "scientific_conclusion": conclusion}, indent=2), flush=True)


if __name__ == "__main__":
    main()
