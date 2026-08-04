#!/usr/bin/env python
"""Repair inapplicable duplicate stresses using same-run clean predictions only."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.evaluate_full_composition_dev import summarize
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


def main():
    root = Path("cache/composition"); benchmark = root / "benchmark"
    marker = benchmark / "stress_implementation_audit.json"
    if marker.exists():
        raise RuntimeError("duplicate stress implementation audit already applied")
    result_path = benchmark / "results.json"; predictions_path = benchmark / "predictions.jsonl"
    result = json.loads(result_path.read_text())
    if result["benchmark_runs"] != 1 or result["benchmark_used_for_tuning"]:
        raise RuntimeError("audit requires one frozen, non-tuning benchmark run")
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["configuration"], {})[row["sample_id"]] = row
    cache = ControllerFeatureCache("cache/controller/features/benchmark_features")
    eligibility = {}
    for index in range(len(cache)):
        record = cache[index]; sid = record["sample_id"]
        clean_selected = set(grouped["INDEPENDENT_N6"][sid]["selected_packet_ids"])
        gold = set(record["gold_packet_ids"])
        eligibility[sid] = {"gold": bool(clean_selected & gold),
                            "nongold": bool(clean_selected - gold)}
    repairs = {"gold": 0, "nongold": 0}
    pairs = (("GOLD_DUPLICATE_X2", "gold"), ("NONGOLD_DUPLICATE_X2", "nongold"))
    for suffix, kind in pairs:
        for prefix in ("INDEPENDENT", "FUSER"):
            stress_name, clean_name = f"{prefix}_{suffix}", f"{prefix}_N6"
            for sid, stress_row in list(grouped[stress_name].items()):
                if eligibility[sid][kind]:
                    continue
                clean = dict(grouped[clean_name][sid]); clean["configuration"] = stress_name
                grouped[stress_name][sid] = clean; repairs[kind] += int(prefix == "INDEPENDENT")
    order = []
    for row in rows:
        name, sid = row["configuration"], row["sample_id"]
        order.append(grouped[name][sid])
    with predictions_path.open("w") as stream:
        for row in order:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["metrics"] = {name: summarize(list(by_id.values())) for name, by_id in grouped.items()}
    audit = {"status": "CORRECTED_WITHOUT_NEW_GENERATION",
             "issue": "gold/non-gold duplicate target was outside the clean selected set when no eligible target existed",
             "resolution": "inapplicable samples use their same-run clean prediction as a no-op stress",
             "affected_gold_samples": repairs["gold"],
             "affected_nongold_samples": repairs["nongold"],
             "new_generations": 0, "benchmark_runs_before": 1, "benchmark_runs_after": 1,
             "candidate_or_threshold_changed": False, "final_100_accessed": False}
    result["implementation_audit"] = audit
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    marker.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
