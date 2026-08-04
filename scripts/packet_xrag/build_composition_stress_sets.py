#!/usr/bin/env python
"""Materialize locked order and duplicate stress-set definitions for diagnostic-150."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.composition.diagnostics import (
    document_order, duplicate_stress_sets, fixed_permutation,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


def main():
    root = Path("cache/composition")
    output = root / "diagnostics/stress_sets.jsonl"
    if output.exists(): raise RuntimeError("refusing to overwrite composition stress sets")
    ids = json.loads((root / "diagnostics/diagnostic_ids.json").read_text())["ordered_sample_ids"]
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    with output.open("w") as stream:
        for sid in ids:
            record = cache[by_id[sid]]; row = {"sample_id": sid, "orders": {}, "duplicates": {}}
            for breadth in (4, 6):
                selected = record["topk_ranking"][:breadth]
                row["orders"][str(breadth)] = {
                    "relevance": selected, "reverse": list(reversed(selected)),
                    "document": document_order(record, selected),
                    **{f"random_{index}": fixed_permutation(selected, sid, index)
                       for index in range(3)}}
            row["duplicates"] = duplicate_stress_sets(record, record["topk_ranking"][:3])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "diagnostics/stress_sets_manifest.json").write_text(json.dumps({
        "status": "complete", "sample_count": len(ids), "seed": 20260804,
        "order_breadths": [4, 6], "order_variants": 6,
        "duplicate_base": "TOPK_3", "duplicate_targets": ["gold", "top_non_gold", "random"],
        "duplicate_counts": [1, 2, 4], "gold_used_for_stress_construction_only": True,
        "final_100_accessed": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "samples": len(ids)}))


if __name__ == "__main__": main()
