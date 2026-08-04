#!/usr/bin/env python
"""Analyze SEARCH_DEV failures before proposing deployable controller features."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.controller.autonomous_search import (
    SubsetFeatureCache, assert_only_search_dev, load_search_split, read_jsonl,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset


def ordered_key(sid, selected, candidate):
    return sid, tuple(selected), int(candidate)


def compact_case(record, static_row, b_row, details):
    return {"sample_id": record["sample_id"], "question": record["question"],
            "static_prediction": static_row.get("short_prediction") if static_row else None,
            "model_b_prediction": b_row.get("short_prediction") if b_row else None,
            "static_selected": static_row.get("selected_packet_ids") if static_row else None,
            "model_b_selected": b_row.get("selected_packet_ids") if b_row else None,
            "details": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--model-b-predictions", default="cache/controller/utility_predictor/model_b/internal_dev_grid_predictions.jsonl")
    args = parser.parse_args(argv); root = Path(args.root)
    output = root / "root_cause_analysis.md"
    if output.exists(): raise RuntimeError("refusing to overwrite root-cause analysis")
    dev, shadow = load_search_split(root); dev_ids = set(dev["ordered_sample_ids"])
    parent = ControllerFeatureCache(args.feature_cache)
    cache = SubsetFeatureCache(parent, dev["ordered_sample_ids"])
    records = {cache[index]["sample_id"]: cache[index] for index in range(len(cache))}
    baseline = read_jsonl(root / "stage0/search_dev_baselines.jsonl")
    assert_only_search_dev([row["sample_id"] for row in baseline], dev_ids, "baseline predictions")
    static = {row["sample_id"]: row for row in baseline if row["configuration"] == "STATIC_2"}
    b_rows = [row for row in read_jsonl(args.model_b_predictions)
              if row["configuration"] == "MODEL_B_E1_TAU0.10" and row["sample_id"] in dev_ids]
    assert_only_search_dev([row["sample_id"] for row in b_rows], dev_ids, "Model-B analysis")
    model_b = {row["sample_id"]: row for row in b_rows}
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    label_rows = [row for row in labels.rows if row["sample_id"] in dev_ids]
    lookup = {ordered_key(row["sample_id"], row["selected_packet_ids"], row["candidate_packet_id"]): row
              for row in label_rows}
    groups = defaultdict(list)
    for row in label_rows: groups[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(row)
    categories = {name: [] for name in (
        "static_correct_model_b_wrong", "static_wrong_oracle_utility_correct",
        "static_selects_harmful_packet", "static_misses_useful_non_gold",
        "model_b_stops_too_early", "model_b_continues_unnecessarily")}
    for sid in dev["ordered_sample_ids"]:
        record, static_row, b_row = records[sid], static[sid], model_b[sid]
        if static_row["short_f1"] == 1.0 and b_row["short_f1"] < 1.0:
            categories["static_correct_model_b_wrong"].append(
                compact_case(record, static_row, b_row, {"static_f1": 1.0, "model_b_f1": b_row["short_f1"]}))
        s0 = groups.get((sid, tuple()), [])
        if static_row["short_f1"] < 1.0 and s0:
            best = max(s0, key=lambda row: (row["delta_utility"], -row["candidate_packet_id"]))
            if best["delta_utility"] > .02 and best["candidate_is_gold"]:
                categories["static_wrong_oracle_utility_correct"].append(
                    compact_case(record, static_row, b_row, {"oracle_packet": best["candidate_packet_id"],
                                "oracle_delta": best["delta_utility"]}))
        harmful = []
        selected_before = []
        for packet_id in static_row["selected_packet_ids"]:
            row = lookup.get(ordered_key(sid, selected_before, packet_id))
            if row is not None and row["delta_utility"] < -.02:
                harmful.append({"packet_id": packet_id, "step": len(selected_before),
                                "delta": row["delta_utility"], "static_score": row["static_score"],
                                "query_cosine": row["query_cosine"]})
            selected_before = selected_before + [packet_id]
        if harmful:
            categories["static_selects_harmful_packet"].append(
                compact_case(record, static_row, b_row, {"harmful_actions": harmful}))
        missed = [row for row in s0 if not row["candidate_is_gold"] and
                  row["candidate_packet_id"] not in static_row["selected_packet_ids"] and
                  row["delta_utility"] > .02]
        if missed:
            best = max(missed, key=lambda row: row["delta_utility"])
            categories["static_misses_useful_non_gold"].append(
                compact_case(record, static_row, b_row, {"packet_id": best["candidate_packet_id"],
                            "delta": best["delta_utility"], "source_tags": best["candidate_source_tags"]}))
        stop_group = groups.get((sid, tuple(b_row["selected_packet_ids"])), [])
        if stop_group:
            best = max(stop_group, key=lambda row: row["delta_utility"])
            if best["delta_utility"] > .02:
                categories["model_b_stops_too_early"].append(
                    compact_case(record, static_row, b_row, {"best_remaining": best["candidate_packet_id"],
                                "delta": best["delta_utility"]}))
        unnecessary = []
        for action in b_row.get("actions", []):
            state_group = groups.get((sid, tuple(action["selected_packet_ids_before"])), [])
            chosen = lookup.get(ordered_key(sid, action["selected_packet_ids_before"], action["packet_id"]))
            if state_group and max(row["delta_utility"] for row in state_group) <= .02:
                unnecessary.append({"step": action["step"], "packet_id": action["packet_id"],
                                    "chosen_delta": chosen["delta_utility"] if chosen else None})
        if unnecessary:
            categories["model_b_continues_unnecessarily"].append(
                compact_case(record, static_row, b_row, {"actions": unnecessary}))
    for name in categories: categories[name] = categories[name][:50]
    candidate_states = defaultdict(list)
    for row in label_rows:
        candidate_states[(row["sample_id"], row["candidate_packet_id"])].append(row["delta_utility"])
    sign_flips = sum(min(values) < -.02 and max(values) > .02 for values in candidate_states.values())
    sufficient_non_gold = [row for row in label_rows if row["state_full_gold_support"] and
                           not row["candidate_is_gold"]]
    sufficient_harm = (sum(row["delta_utility"] < -.02 for row in sufficient_non_gold) /
                       len(sufficient_non_gold))
    findings = {
        "category_counts_capped_at_50": {key: len(value) for key, value in categories.items()},
        "search_dev_candidate_sign_flip_fraction": sign_flips / len(candidate_states),
        "sufficient_state_non_gold_harmful_fraction": sufficient_harm,
        "interpretation": [
            "STATIC-2 is strong on SEARCH_DEV but still sometimes commits a truly harmful early action.",
            "Useful non-gold packets exist outside STATIC-2, so support classification alone is insufficient.",
            "Model B exhibits both premature STOP and unnecessary continuation on auditable states.",
            "The dominant missing information is generator answer sufficiency and answer-relative candidate effect.",
            "Pooled SFR similarity should remain a routing prior, not the sole STOP or utility signal.",
        ],
        "family_decisions": {
            "A_generator_state_stop": "KEEP: directly observes answer confidence/sufficiency.",
            "B_answer_conditioned_reranking": "KEEP: targets useful non-gold and conflict/repetition patterns; audit self-confirmation.",
            "C_candidate_intervention": "KEEP AS COSTLY PROBE: directly measures generator response under deployable constraints.",
            "D_text_cross_encoder": "KEEP LIGHTWEIGHT PROBE: tests information lost by pooled embeddings.",
            "E_direct_stop_static": "KEEP AS PRIMARY SIMPLE POLICY: isolates STOP from packet ranking.",
            "F_hybrid": "DEFER UNTIL COMPONENT PROBES: only combine independently useful signals.",
        },
    }
    analysis_json = {"search_dev_hash": dev["sha256"], "search_shadow_accessed": False,
                     "findings": findings, "categories": categories,
                     "final_100_accessed": False, "final_100_runs": 0}
    (root / "root_cause_analysis.json").write_text(
        json.dumps(analysis_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    lines = ["# Autonomous Controller Root-Cause Analysis", "",
             f"- SEARCH_DEV hash: `{dev['sha256']}`", "- SEARCH_SHADOW accessed: No", "",
             "## Main findings", ""]
    lines.extend(f"- {item}" for item in findings["interpretation"])
    lines.extend(["", f"- Candidate sign-flip fraction: {findings['search_dev_candidate_sign_flip_fraction']:.6f}",
                  f"- Sufficient-state non-gold harmful fraction: {sufficient_harm:.6f}", "",
                  "## Failure categories", ""])
    for name, items in categories.items():
        lines.append(f"- {name}: {len(items)} retained examples")
    lines.extend(["", "## Family decisions", ""])
    lines.extend(f"- {name}: {decision}" for name, decision in findings["family_decisions"].items())
    lines.extend(["", "Detailed capped examples are stored in `root_cause_analysis.json`.",
                  "", "- Final 100 accessed: No", "- Final 100 runs: 0", ""])
    output.write_text("\n".join(lines))
    print(json.dumps(findings, indent=2), flush=True)


if __name__ == "__main__": main()
