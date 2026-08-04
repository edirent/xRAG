#!/usr/bin/env python
"""Run the final consolidated, non-selective DEV mechanism ablation suite."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector,
)
from scripts.packet_xrag.evaluate_full_composition_dev import (
    evaluate_fuser, sha256, summarize,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


class ResidualAblation(nn.Module):
    def __init__(self, fuser, zero_query=False, zero_base=False):
        super().__init__(); self.fuser = fuser
        self.zero_query = zero_query; self.zero_base = zero_base

    def forward(self, query, base, extras, mask):
        if self.zero_query:
            query = torch.zeros_like(query)
        if self.zero_base:
            base = torch.zeros_like(base)
        return self.fuser(query, base, extras, mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root); output_dir = root / "ablations"
    if output_dir.exists():
        raise RuntimeError("refusing to overwrite candidate mechanism ablations")
    shadow = json.loads((root / "shadow/results.json").read_text())
    benchmark = json.loads((root / "benchmark/bootstrap.json").read_text())
    if shadow["status"] != "PASS" or benchmark["status"] != "PASS":
        raise RuntimeError("candidate ablations require a frozen successful candidate")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    if ledger["usage"]["composition_dev_generation"] != 9:
        raise RuntimeError("final ablation suite must consume DEV generation evaluation 10/10")
    candidate = json.loads((root / "frozen_candidate/candidate_config.json").read_text())
    if sha256(candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
        raise RuntimeError("candidate checkpoint hash mismatch")
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    ids = json.loads((root / "splits/composition_dev_ids.json").read_text())["ordered_sample_ids"]
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    records = [cache[by_id[sid]] for sid in ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/train_static_scores.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index)) for record in records}
    static_groups = [rankings[record["sample_id"]][:6] for record in records]
    topk_groups = [record["topk_ranking"][:6] for record in records]
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device)
    payload = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
    fuser = build_fuser("C1").to(device); fuser.load_state_dict(payload["state_dict"], strict=True)
    fuser.eval(); rows_by_name = {}
    configurations = (
        ("QUERY_AGNOSTIC_STATIC_N6", ResidualAblation(fuser, zero_query=True), static_groups),
        ("NO_RESIDUAL_BASE_STATIC_N6", ResidualAblation(fuser, zero_base=True), static_groups),
        ("TOPK_PRESELECTION_N6", fuser, topk_groups),
    )
    for name, model, groups in configurations:
        model.eval(); rows = evaluate_fuser(name, model, records, groups, k2, tokenizer,
            generator, xrag_id, device, args.batch_size); rows_by_name[name] = rows
        print(json.dumps({name: summarize(rows)}), flush=True)
    metrics = {name: summarize(rows) for name, rows in rows_by_name.items()}
    full_dev = json.loads((root / "dev_full/results.json").read_text())
    reference = full_dev["metrics"]["FUSER_C1_O1_N6"]
    result = {"status": "COMPLETE_NON_SELECTIVE", "split": "COMPOSITION_DEV",
              "reference_static_query_residual_n6": reference, "metrics": metrics,
              "deltas_vs_reference": {name: value["short_f1"] - reference["short_f1"]
                                      for name, value in metrics.items()},
              "candidate_changed": False, "thresholds_changed": False,
              "benchmark_reused_for_selection": False, "shadow_accessed_again": False,
              "benchmark_accessed_again": False, "final_100_accessed": False}
    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "predictions.jsonl").open("w") as stream:
        for rows in rows_by_name.values():
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger["usage"]["composition_dev_generation"] += 1
    ledger["final_candidate_ablations"] = {"status": result["status"],
        "configurations": list(metrics), "candidate_changed": False}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
