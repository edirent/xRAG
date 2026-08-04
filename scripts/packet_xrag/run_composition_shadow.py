#!/usr/bin/env python
"""Freeze the sole DEV winner, then evaluate COMPOSITION_SHADOW exactly once."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector,
)
from scripts.packet_xrag.evaluate_full_composition_dev import (
    evaluate_fuser, evaluate_independent, paired_bootstrap, reduction, robustness,
    sha256, stress_groups, summarize,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.composition.protocol import assert_evaluation_lock
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


def freeze_candidate(root, dev):
    candidate = dev["selected_candidate"]
    if not candidate or candidate not in dev["passing_models"]:
        raise RuntimeError("DEV did not produce a unique promotable candidate")
    frozen = root / "frozen_candidate"
    if frozen.exists():
        raise RuntimeError("refusing to overwrite frozen composition candidate")
    model = dev["models"][candidate]
    config = {"run": candidate, "architecture": "Residual STATIC2 + extra-evidence fusion",
              "branch": "C1", "checkpoint": model["checkpoint"],
              "checkpoint_sha256": model["checkpoint_sha256"],
              "input_selector": "STATIC", "input_breadth": 6, "output_slots": 4,
              "objectives": ["O1"] if candidate == "C1_O1" else ["O1", "O3", "O4"],
              "packet_ordering": "descending STATIC score; packet-index tie break",
              "dev_gates": model["gates"], "thresholds": {
                  "shadow_quality_gain": 1.5, "shadow_composition_gain": 4.0,
                  "shadow_static_tolerance": 0.5, "shadow_breadth_drop": 1.0,
                  "shadow_robustness_reduction": 0.4},
              "seed": 20260804, "shadow_tuning_allowed": False,
              "benchmark_tuning_allowed": False, "final_100_accessed": False}
    frozen.mkdir(parents=True)
    (frozen / "candidate_config.json").write_text(json.dumps(config, indent=2,
                                                               sort_keys=True) + "\n")
    manifest = {"checkpoint": model["checkpoint"], "sha256": model["checkpoint_sha256"],
                "exists": Path(model["checkpoint"]).is_file(),
                "checkpoint_selection_split": "COMPOSITION_DEV",
                "checkpoint_selection_rule": dev["selection_rule"]}
    (frozen / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                                  sort_keys=True) + "\n")
    (frozen / "checkpoint_hashes.json").write_text(json.dumps({
        "fuser": model["checkpoint_sha256"],
        "k2": sha256("cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt"),
        "static_scorer": sha256("cache/controller/static/best_short_f1/scorer.pt"),
        "generator_projector": sha256("cache/projector/packet_projector_calibration/last/projector.pt")
    }, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(root / "dev_full/results.json", frozen / "dev_results.json")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv); root = Path(args.root)
    output_dir = root / "shadow"; output = output_dir / "results.json"
    if output_dir.exists():
        raise RuntimeError("refusing to overwrite composition shadow results")
    dev = json.loads((root / "dev_full/results.json").read_text())
    if dev["status"] != "PASS":
        raise RuntimeError("Shadow remains sealed because DEV gate did not pass")
    config = freeze_candidate(root, dev)
    lock_path = root / "composition_shadow_lock.json"
    lock = assert_evaluation_lock(lock_path, "COMPOSITION_SHADOW", 2)
    # Consume the run before unsealing IDs. A crash therefore cannot silently retry.
    lock["runs"] += 1; lock["candidate_checkpoint_sha256"] = config["checkpoint_sha256"]
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    shadow_ids = json.loads((root / "splits/composition_shadow_ids.json").read_text())["ordered_sample_ids"]
    cache = ControllerFeatureCache("cache/controller/features/train_features")
    by_id = {record["sample_id"]: index for index, record in enumerate(cache.records)}
    records = [cache[by_id[sid]] for sid in shadow_ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache, "cache/controller/static/best_short_f1/scorer.pt",
        "cache/controller/utility_predictor/features/train_static_scores.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index)) for record in records}
    tokenizer, generator, xrag_id, model_config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(model_config, device)
    payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise RuntimeError("frozen candidate checkpoint hash mismatch")
    fuser = build_fuser("C1").to(device); fuser.load_state_dict(payload["state_dict"], strict=True)
    fuser.eval(); stresses = stress_groups(records, rankings); rows_by_name = {}; all_rows = []

    def add(name, rows):
        rows_by_name[name] = rows; all_rows.extend(rows)
        print(json.dumps({name: summarize(rows)}), flush=True)

    static_groups = [rankings[record["sample_id"]][:2] for record in records]
    add("STATIC_2", evaluate_independent("STATIC_2", records, static_groups, k2,
        tokenizer, generator, xrag_id, device, args.batch_size))
    add("INDEPENDENT_N6", evaluate_independent("INDEPENDENT_N6", records,
        stresses["CLEAN"], k2, tokenizer, generator, xrag_id, device, args.batch_size))
    add("FUSER_N6", evaluate_fuser("FUSER_N6", fuser, records, stresses["CLEAN"], k2,
        tokenizer, generator, xrag_id, device, args.batch_size))
    for name in ("REVERSE", "RANDOM_0", "RANDOM_1", "RANDOM_2", "DUPLICATE_X2",
                 "DISTRACTOR_X4"):
        label = f"ORDER_{name}" if name.startswith(("REVERSE", "RANDOM")) else name
        add(f"INDEPENDENT_{label}", evaluate_independent(f"INDEPENDENT_{label}", records,
            stresses[name], k2, tokenizer, generator, xrag_id, device, args.batch_size))
        add(f"FUSER_{label}", evaluate_fuser(f"FUSER_{label}", fuser, records,
            stresses[name], k2, tokenizer, generator, xrag_id, device, args.batch_size))
    metrics = {name: summarize(rows) for name, rows in rows_by_name.items()}
    independent_robustness = robustness(metrics, "INDEPENDENT")
    fuser_robustness = robustness(metrics, "FUSER")
    reductions = {name: reduction(independent_robustness, fuser_robustness, name)
                  for name in ("order_degradation", "duplicate_degradation",
                               "distractor_degradation")}
    bootstrap = paired_bootstrap(rows_by_name["FUSER_N6"], rows_by_name["STATIC_2"])
    static2, independent, fused = (metrics["STATIC_2"]["short_f1"],
        metrics["INDEPENDENT_N6"]["short_f1"], metrics["FUSER_N6"]["short_f1"])
    breadth_drop = static2 - fused  # C1 N2 is exactly STATIC2 by construction.
    gates = {"shadow_quality": fused - static2 >= 1.5 and bootstrap["p_delta_gt_0"] >= .9,
             "shadow_composition": fused >= static2 - .5 and fused - independent >= 4 and
                                   breadth_drop <= 1,
             "shadow_robustness": fused >= static2 - .5 and
                                  sum(value >= .4 for value in reductions.values()) >= 2}
    status = "PASS" if any(gates.values()) else "MANDATORY_STOP_SHADOW_FAILED"
    result = {"status": status, "split": "COMPOSITION_SHADOW", "sample_count": len(records),
              "candidate_config": config, "metrics": metrics,
              "paired_bootstrap_fuser_vs_static2": bootstrap,
              "breadth_drop_n2_to_n6": breadth_drop,
              "independent_robustness": independent_robustness,
              "fuser_robustness": fuser_robustness, "robustness_reductions": reductions,
              "gates": gates, "shadow_run": lock["runs"], "benchmark_accessed": False,
              "final_100_accessed": False}
    output_dir.mkdir(parents=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "predictions.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["usage"]["composition_shadow_evaluations"] += 1
    ledger["stage7_shadow"] = {"status": status, "gates": gates,
                               "checkpoint_hash": config["checkpoint_sha256"]}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "gates": gates, "metrics": metrics,
                      "robustness_reductions": reductions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
