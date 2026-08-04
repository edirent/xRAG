#!/usr/bin/env python
"""Create the locked search split, audit, shadow seal, and experiment ledger."""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag.token_resampler_common import sha256_file
from src.packet_xrag.controller.autonomous_search import (
    EXPECTED_BENCHMARK_HASH, EXPECTED_INTERNAL_DEV_HASH, EXPECTED_TRAIN_HASH,
    QUARANTINED_ID, SEARCH_SEED, fixed_probe_subset, ordered_ids_sha256,
    split_search_ids,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache


EXPECTED = {
    "v1": "fa1a9ba443960acc176dc387989fe7c2e3fa1ef1cf24a38db265da4c1c60f760",
    "k2": "c40aa3dc297f57b5be73649f1754dc292ef98fefbc5c8a338b90e108427fd8a4",
    "static": "9ea9609ba1fd6ab1466610c2324aa1938d86604c6b23c07b0e68ca831ccb8276",
}


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def branch(branch_id, family, hypothesis, failure, inputs, supervision, expected,
           falsification, cost, promotion):
    return {"experiment_id": branch_id, "family": family, "hypothesis": hypothesis,
            "previous_failure_addressed": failure, "deployable_inference_inputs": inputs,
            "training_supervision": supervision, "expected_result": expected,
            "falsification_criterion": falsification, "estimated_cost": cost,
            "promotion_criterion": promotion, "status": "preregistered",
            "actual_metrics": None, "result": None, "decision": None, "reason": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    parser.add_argument("--internal-dev-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--train-cache", default="cache/controller/features/train_features")
    parser.add_argument("--benchmark-cache", default="cache/controller/features/benchmark_features")
    parser.add_argument("--utility-audit", default="cache/controller/utility_predictor/checkpoint_audit.json")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    args = parser.parse_args(argv); root = Path(args.root)
    if (root / "splits/search_shadow_ids.json").exists():
        raise RuntimeError("refusing to overwrite the sealed autonomous-search split")
    dev_cache = ControllerFeatureCache(args.internal_dev_cache)
    train_cache = ControllerFeatureCache(args.train_cache)
    benchmark_cache = ControllerFeatureCache(args.benchmark_cache)
    if dev_cache.manifest["effective_split_hash"] != EXPECTED_INTERNAL_DEV_HASH:
        raise RuntimeError("internal-dev hash mismatch")
    if train_cache.manifest["effective_split_hash"] != EXPECTED_TRAIN_HASH:
        raise RuntimeError("train hash mismatch")
    if benchmark_cache.manifest["effective_split_hash"] != EXPECTED_BENCHMARK_HASH:
        raise RuntimeError("benchmark hash mismatch")
    ordered = [record["sample_id"] for record in dev_cache.records]
    search_dev, search_shadow = split_search_ids(ordered)
    probe = fixed_probe_subset(search_dev)
    common = {"parent_split": "controller_internal_dev_500",
              "parent_split_hash": EXPECTED_INTERNAL_DEV_HASH, "seed": SEARCH_SEED}
    write_json(root / "splits/search_dev_ids.json", {
        **common, "split": "SEARCH_DEV", "sample_count": len(search_dev),
        "ordered_sample_ids": search_dev, "sha256": ordered_ids_sha256(search_dev)})
    write_json(root / "splits/search_shadow_ids.json", {
        **common, "split": "SEARCH_SHADOW", "sample_count": len(search_shadow),
        "ordered_sample_ids": search_shadow, "sha256": ordered_ids_sha256(search_shadow)})
    write_json(root / "splits/probe_subset_ids.json", {
        **common, "split": "SEARCH_DEV_PROBE_150", "sample_count": len(probe),
        "ordered_sample_ids": probe, "sha256": ordered_ids_sha256(probe)})
    split_audit = {"status": "PASS", "seed": SEARCH_SEED,
                   "search_dev_count": len(search_dev), "search_shadow_count": len(search_shadow),
                   "overlap": [], "union_matches_internal_dev": set(search_dev + search_shadow) == set(ordered),
                   "quarantined_id_present": QUARANTINED_ID in set(search_dev + search_shadow),
                   "search_dev_hash": ordered_ids_sha256(search_dev),
                   "search_shadow_hash": ordered_ids_sha256(search_shadow),
                   "probe_subset_hash": ordered_ids_sha256(probe),
                   "shadow_generation_evaluations": 0, "benchmark_evaluations": 0,
                   "final_100_accessed": False, "final_100_runs": 0}
    if split_audit["quarantined_id_present"] or not split_audit["union_matches_internal_dev"]:
        raise RuntimeError("search split isolation failure")
    write_json(root / "splits/split_audit.json", split_audit)
    prior = json.loads(Path(args.utility_audit).read_text())
    static_hash = sha256_file(args.static_checkpoint)
    if prior["status"] != "PASS" or prior["v1_sha256"] != EXPECTED["v1"] or \
            prior["k2_sha256"] != EXPECTED["k2"] or static_hash != EXPECTED["static"]:
        raise RuntimeError("frozen checkpoint compatibility failure")
    checkpoint = {"status": "PASS", "source_formal_audit": str(Path(args.utility_audit).resolve()),
                  "source_formal_audit_sha256": sha256_file(args.utility_audit),
                  "v1_sha256": EXPECTED["v1"], "k2_sha256": EXPECTED["k2"],
                  "static_sha256": static_hash, "effective_train_hash": EXPECTED_TRAIN_HASH,
                  "internal_dev_hash": EXPECTED_INTERNAL_DEV_HASH,
                  "search_dev_hash": split_audit["search_dev_hash"],
                  "search_shadow_hash": split_audit["search_shadow_hash"],
                  "benchmark_hash": EXPECTED_BENCHMARK_HASH,
                  "prompt_hash": prior["prompt_hash"],
                  "answer_extractor_hash": hashlib.sha256(
                      inspect.getsource(selector.extract_short_answer).encode()).hexdigest(),
                  "generator_frozen": prior["trainable_generator_parameters"] == 0,
                  "sfr_frozen": prior["trainable_sfr_parameters"] == 0,
                  "k2_frozen": prior["trainable_k2_parameters"] == 0,
                  "static_frozen": prior["trainable_static_parameters"] == 0,
                  "loaded_adapters": prior["loaded_adapters"],
                  "final_100_accessed": False, "final_100_runs": 0}
    if not all(checkpoint[key] for key in
               ("generator_frozen", "sfr_frozen", "k2_frozen", "static_frozen")):
        raise RuntimeError("frozen parameter audit failed")
    write_json(root / "checkpoint_audit.json", checkpoint)
    (root / "checkpoint_audit.md").write_text(
        "# Autonomous Controller Checkpoint Audit\n\n- Status: PASS\n"
        f"- V1: `{EXPECTED['v1']}`\n- K2: `{EXPECTED['k2']}`\n"
        f"- STATIC: `{EXPECTED['static']}`\n- SEARCH_DEV: `{split_audit['search_dev_hash']}`\n"
        f"- SEARCH_SHADOW: `{split_audit['search_shadow_hash']}`\n"
        "- Generator/SFR/K2/STATIC frozen: Yes\n- Final 100 accessed: No\n")
    write_json(root / "search_shadow_seal.json", {
        "status": "SEALED", "allowed_after": "unique_final_candidate_frozen",
        "evaluation_runs": 0, "maximum_runs": 2, "sample_hash": split_audit["search_shadow_hash"]})
    branches = [
        branch("B1", "A/E", "Generator confidence exposes selected-set sufficiency for direct STOP.",
               "Embedding state shift learned state identity but not reliable STOP.",
               ["query", "selected packets", "provisional answer token confidence/entropy"],
               "gold-utility STOP/CONTINUE labels", "STOP AUROC >= 0.70",
               "AUROC < 0.70 and AUPRC gain < 0.10", "one frozen generation per state",
               "Stage-1 branch gate"),
        branch("B2", "B", "Provisional-answer consistency distinguishes supporting, duplicate, and conflicting candidates.",
               "Pooled candidate features cannot model answer-relative evidence.",
               ["question", "selected texts", "candidate text", "provisional answer"],
               "utility ranking/sign labels", "within-state Spearman >= 0.20",
               "no candidate gate improves over retriever MLP", "text features; no candidate generation",
               "Stage-1 branch gate"),
        branch("B3", "C", "Candidate intervention confidence changes predict corrective evidence.",
               "Static relevance cannot observe generator response to candidate addition.",
               ["provisional answer", "STATIC top-6", "self-answer likelihood under candidate intervention"],
               "utility ranking/sign labels", "regret reduction >= 25%",
               "regret reduction < 25%", "up to four probes/step; <=12 forwards/sample",
               "Stage-1 branch gate"),
        branch("B4", "D", "Text-level lexical relation features recover information discarded by pooled SFR embeddings.",
               "Candidate utility Spearman remained weak with pooled vectors.",
               ["question text", "selected packet texts", "candidate packet text"],
               "utility ranking/sign labels", "best-action top-1 gain >= 8pp",
               "no candidate gate reached", "lightweight sparse text model",
               "Stage-1 branch gate"),
        branch("B5", "A", "Prompt/generation state plus packet count predicts whether evidence is sufficient.",
               "Global learned state shift lacks generator-internal sufficiency evidence.",
               ["generator token statistics", "packet count", "STATIC prefix statistics"],
               "gold-utility STOP/CONTINUE labels", "STOP AUPRC gain >= 0.10",
               "AUPRC gain < 0.10 and AUROC < 0.70", "one generator pass/state",
               "Stage-1 branch gate"),
        branch("B6", "F", "A small calibrated ensemble combines complementary STOP and answer-relative signals.",
               "Single feature families may be individually noisy.",
               ["STATIC", "generator confidence", "answer consistency", "packet-count prior"],
               "STOP and action labels", "one probe gate and improved calibration",
               "no branch gate reached", "sum of promoted lightweight features",
               "Stage-1 branch gate"),
    ]
    ledger = {"protocol": "PacketRAG Autonomous Controller Discovery", "seed": SEARCH_SEED,
              "budgets": {"hypothesis_branches_max": 6, "feasibility_probes_max": 8,
                          "full_training_runs_max": 4, "search_dev_generation_max": 8,
                          "search_shadow_evaluations_max": 2, "benchmark_evaluations_max": 1},
              "usage": {"hypothesis_branches": 6, "feasibility_probes": 0,
                        "full_training_runs": 0, "search_dev_generation": 0,
                        "search_shadow_evaluations": 0, "benchmark_evaluations": 0},
              "branches": branches, "final_100_accessed": False, "final_100_runs": 0}
    write_json(root / "experiment_ledger.json", ledger)
    lines = ["# Autonomous Controller Experiment Ledger", "",
             "- SEARCH_SHADOW: SEALED", "- Benchmark runs: 0", "- Final 100 runs: 0", ""]
    for item in branches:
        lines.extend([f"## {item['experiment_id']} — Family {item['family']}", "",
                      f"- Hypothesis: {item['hypothesis']}", f"- Inputs: {', '.join(item['deployable_inference_inputs'])}",
                      f"- Falsification: {item['falsification_criterion']}", "- Status: preregistered", ""])
    (root / "experiment_ledger.md").write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "search_dev_hash": split_audit["search_dev_hash"],
                      "search_shadow_hash": split_audit["search_shadow_hash"],
                      "probe_subset_hash": split_audit["probe_subset_hash"]}, indent=2))


if __name__ == "__main__": main()

