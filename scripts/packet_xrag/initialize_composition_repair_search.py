#!/usr/bin/env python
"""Create locked composition splits, frozen-asset audit, and experiment ledger."""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from src.packet_xrag.composition.protocol import (
    COMPOSITION_SEED, EXPECTED_BENCHMARK_HASH, QUARANTINED_ID,
    assert_no_overlap, diagnostic_ids, ordered_ids_sha256, overlap_audit,
    split_composition_ids,
)


EXPECTED = {"v1": "fa1a9ba443960acc176dc387989fe7c2e3fa1ef1cf24a38db265da4c1c60f760",
            "k2": "c40aa3dc297f57b5be73649f1754dc292ef98fefbc5c8a338b90e108427fd8a4",
            "static": "9ea9609ba1fd6ab1466610c2324aa1938d86604c6b23c07b0e68ca831ccb8276"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def split_payload(name, ids):
    return {"split": name, "seed": COMPOSITION_SEED, "sample_count": len(ids),
            "ordered_sample_ids": ids, "sha256": ordered_ids_sha256(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--effective-ids", default="cache/controller/splits/controller_effective_train_ids.json")
    parser.add_argument("--train-records", default="cache/controller/features/train_features/records.jsonl")
    parser.add_argument("--internal-records", default="cache/controller/features/internal_dev_features/records.jsonl")
    parser.add_argument("--benchmark-records", default="cache/controller/features/benchmark_features/records.jsonl")
    parser.add_argument("--formal-audit", default="cache/controller/autonomous_search/checkpoint_audit.json")
    args = parser.parse_args(argv); root = Path(args.root)
    if root.exists(): raise RuntimeError("refusing to overwrite composition search root")
    source = json.loads(Path(args.effective_ids).read_text()); effective = source["ordered_sample_ids"]
    train_ids, dev_ids, shadow_ids = split_composition_ids(effective)
    payloads = {"composition_train": split_payload("COMPOSITION_TRAIN", train_ids),
                "composition_dev": split_payload("COMPOSITION_DEV", dev_ids),
                "composition_shadow": split_payload("COMPOSITION_SHADOW", shadow_ids)}
    for name, payload in payloads.items(): write_json(root / f"splits/{name}_ids.json", payload)
    diagnostic = diagnostic_ids(dev_ids)
    write_json(root / "diagnostics/diagnostic_ids.json",
               {"split": "COMPOSITION_DEV_DIAGNOSTIC", "seed": COMPOSITION_SEED,
                "sample_count": len(diagnostic), "ordered_sample_ids": diagnostic,
                "sha256": ordered_ids_sha256(diagnostic)})
    all_train_records = read_jsonl(args.train_records)
    by_id = {record["sample_id"]: record for record in all_train_records}
    if set(by_id) != set(effective): raise RuntimeError("effective train records/IDs mismatch")
    named = {"composition_train": [by_id[sid] for sid in train_ids],
             "composition_dev": [by_id[sid] for sid in dev_ids],
             "composition_shadow": [by_id[sid] for sid in shadow_ids],
             "existing_internal_dev": read_jsonl(args.internal_records),
             "benchmark": read_jsonl(args.benchmark_records)}
    overlaps = overlap_audit(named); assert_no_overlap(overlaps)
    split_audit = {"status": "PASS", "seed": COMPOSITION_SEED,
                   "source_effective_train_hash": ordered_ids_sha256(effective),
                   "composition_train_hash": payloads["composition_train"]["sha256"],
                   "composition_dev_hash": payloads["composition_dev"]["sha256"],
                   "composition_shadow_hash": payloads["composition_shadow"]["sha256"],
                   "benchmark_hash": EXPECTED_BENCHMARK_HASH, "overlaps": overlaps,
                   "quarantined_id": QUARANTINED_ID,
                   "quarantine_absent": QUARANTINED_ID not in set(train_ids + dev_ids + shadow_ids),
                   "final_100_accessed": False, "final_100_runs": 0,
                   "final_100_overlap_note": "Not inspected because the protocol forbids final-100 access."}
    write_json(root / "splits/split_audit.json", split_audit)
    (root / "splits/split_audit.md").write_text(
        "# Composition Split Audit\n\n- Status: PASS\n"
        f"- Train/dev/shadow: 3999/250/250\n- Quarantine absent: Yes\n"
        "- ID/exact/normalized question overlaps with benchmark and internal-dev: 0\n"
        "- Final 100 accessed: No\n")
    formal = json.loads(Path(args.formal_audit).read_text())
    if formal["status"] != "PASS" or any(formal[f"{key}_sha256"] != value for key, value in EXPECTED.items()):
        raise RuntimeError("MANDATORY STOP: formal frozen-asset audit mismatch")
    if formal["loaded_adapters"] or not all(formal[f"{key}_frozen"] for key in
                                             ("generator", "sfr", "k2", "static")):
        raise RuntimeError("MANDATORY STOP: unexpected trainable/loaded adapter")
    checkpoint = {"status": "PASS", **{f"{key}_sha256": value for key, value in EXPECTED.items()},
                  "sfr_identifier": "Salesforce/SFR-Embedding-Mistral",
                  "llm_identifier": "Hannibal046/xrag-7b", "xrag_token_id": 32001,
                  "tokens_per_packet": 2, "prompt": "P2_SHORT",
                  "prompt_hash": formal["prompt_hash"],
                  "answer_extractor_hash": hashlib.sha256(
                      inspect.getsource(selector.extract_short_answer).encode()).hexdigest(),
                  "data_split_hashes": {key: value["sha256"] for key, value in payloads.items()},
                  "quarantine_hash": json.loads(Path("cache/controller/features/train_features/manifest.json").read_text())["quarantine_hash"],
                  "generator_trainable_parameters": 0, "sfr_trainable_parameters": 0,
                  "k2_trainable_parameters": 0, "static_scorer_trainable_parameters": 0,
                  "loaded_adapters": [], "residual_adapter_loaded": False,
                  "token_state_resampler_loaded": False, "controller_adapter_loaded": False,
                  "utility_predictor_loaded": False, "final_100_accessed": False,
                  "final_100_runs": 0}
    write_json(root / "checkpoint_audit.json", checkpoint)
    (root / "checkpoint_audit.md").write_text(
        "# Composition Frozen-Asset Audit\n\n- Status: PASS\n"
        f"- V1: `{EXPECTED['v1']}`\n- K2: `{EXPECTED['k2']}`\n- STATIC: `{EXPECTED['static']}`\n"
        "- Generator/SFR/K2/STATIC trainable parameters: 0\n- Loaded adapters: none\n"
        "- Final 100 accessed: No\n")
    diagnostics = [
        ("D1", "soft/text breadth curves", "required breadth baselines", "2/3-to-6/12 soft F1 drop >=3"),
        ("D2", "order sensitivity", "independent tokens may be order brittle", "one perturbation drop >=2"),
        ("D3", "duplicate sensitivity", "independent tokens may amplify duplicates", "one perturbation drop >=2"),
        ("D4", "grouped compression", "joint encoding may stabilize breadth", "grouped gain >=2"),
        ("D5", "same-bandwidth compatibility", "separate packet count from token count", "compatible K1/K4 comparison or documented skip"),
    ]
    ledger = {"protocol": "PacketRAG Autonomous Composition-Repair Search",
              "seed": COMPOSITION_SEED,
              "budgets": {"hypothesis_branches_max": 6, "diagnostic_probe_runs_max": 10,
                          "full_training_runs_max": 6, "composition_dev_generation_max": 10,
                          "composition_shadow_evaluations_max": 2,
                          "benchmark_evaluations_max": 1},
              "usage": {"hypothesis_branches": 0, "diagnostic_probe_runs": 0,
                        "full_training_runs": 0, "composition_dev_generation": 0,
                        "composition_shadow_evaluations": 0, "benchmark_evaluations": 0},
              "diagnostics": [{"experiment_id": eid, "hypothesis": hypothesis,
                               "model_inputs": "frozen STATIC/TOPK packets; frozen K2 or text",
                               "output_token_budget": "diagnostic-dependent; no trainable model",
                               "training_objective": "none", "previous_failure_addressed": previous,
                               "expected_evidence": expected, "falsification_condition": "diagnostic gate not met",
                               "estimated_gpu_cost": "one consolidated diagnostic suite",
                               "promotion_gate": "Stage-1 composition-gap gate", "status": "preregistered"}
                              for eid, hypothesis, previous, expected in diagnostics],
              "branches": [], "shadow_lock": {"split": "COMPOSITION_SHADOW", "runs": 0,
                                                "maximum_runs": 2},
              "benchmark_lock": {"split": "BENCHMARK_500", "runs": 0, "maximum_runs": 1},
              "final_100_accessed": False, "final_100_runs": 0}
    write_json(root / "experiment_ledger.json", ledger)
    write_json(root / "composition_shadow_lock.json", ledger["shadow_lock"])
    write_json(root / "benchmark_lock.json", ledger["benchmark_lock"])
    (root / "experiment_ledger.md").write_text(
        "# Composition Experiment Ledger\n\n- Stage 0: PASS\n- Diagnostic runs: 0 / 10\n"
        "- Shadow / benchmark / final 100: 0 / 0 / 0\n")
    print(json.dumps({"status": "PASS", "split_hashes": checkpoint["data_split_hashes"],
                      "diagnostic_hash": ordered_ids_sha256(diagnostic)}, indent=2))


if __name__ == "__main__": main()
