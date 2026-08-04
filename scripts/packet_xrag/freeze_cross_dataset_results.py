#!/usr/bin/env python
"""Create the immutable cross-dataset freeze package before final-100 access."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_CONFIGS = ("NO_CONTEXT", "TOPK_3", "STATIC_2", "INDEPENDENT_STATIC_6",
                 "FUSER_N6", "FUSER_N12", "INDEPENDENT_ALL", "FUSER_ALL",
                 "XRAG_ORACLE")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    root = REPO_ROOT / "cache/generalization"
    output = root / "final_frozen"
    if output.exists():
        raise RuntimeError("refusing to overwrite unified freeze package")
    gate = json_file(root / "generalization_gate.json")
    if not gate["second_setting_authorized"]:
        raise RuntimeError("generalization gate did not authorize method freezing")
    second = json_file(root / "second_setting/final_results.json")
    ledger = json_file(root / "experiment_ledger.json")
    if ledger["final100_accessed"] or ledger["final100_suite_runs"]:
        raise RuntimeError("final 100 is no longer pristine")
    lock = json_file(root / "final100/lock.json")
    if lock != {"split": "HOTPOT_FINAL100", "runs": 0, "maximum_runs": 1}:
        raise RuntimeError("final-100 access lock is not pristine")
    sealed = json_file(root / "final100/sealed_ids.json")
    if sealed["questions_contexts_answers_accessed"] or sealed["suite_runs"]:
        raise RuntimeError("sealed final-100 manifest records prior access")
    if hashlib.sha256("".join(sealed["ordered_sample_ids"]).encode()).hexdigest() != sealed["sha256"]:
        raise RuntimeError("sealed final-100 ID hash mismatch")
    for dataset in ("2wiki", "musique", "triviaqa"):
        shadow = json_file(root / dataset / "shadow/results.json")
        benchmark = root / dataset / "benchmark/results.json"
        if shadow["gate"]["passed"] != benchmark.exists():
            raise RuntimeError(f"{dataset} benchmark presence disagrees with frozen SHADOW gate")
        if benchmark.exists() and json_file(benchmark)["benchmark_suite_runs"] != 1:
            raise RuntimeError(f"{dataset} benchmark is not a single complete suite")
    leakage = subprocess.run([sys.executable,
        str(REPO_ROOT / "scripts/packet_xrag/audit_cross_dataset_leakage.py")],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if leakage.returncode:
        raise RuntimeError(f"cross-dataset leakage audit failed: {leakage.stdout}{leakage.stderr}")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/packet_xrag"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(f"test gate failed: {tests.stdout}{tests.stderr}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status:
        raise RuntimeError("working tree must be clean before unified freeze")

    global_audit = json_file(root / "global_checkpoint_audit.json")
    asset = json_file(root / "second_setting/asset_selection.json")
    order = json_file(root / "order/O2_evaluation/results.json")
    method_spec = {
        "architecture": "Residual STATIC2 + Extra-Evidence Fusion",
        "base_selector": "STATIC", "base_packets": [1, 2],
        "extra_packets": "STATIC rank 3-N", "main_input_breadth": 6,
        "training_breadths": [2, 4, 6], "output_M": 4,
        "order_policy": "original canonical STATIC order",
        "order_variant_adopted": False, "generator": "frozen", "SFR": "frozen",
        "K2": "frozen", "objective": "frozen answer loss",
        "optimizer": "AdamW", "learning_rate": 5e-5, "weight_decay": .01,
        "warmup_ratio": .05, "epochs": 6, "seed": 20260804,
        "all_packet_cap": 48, "controller_route_reopened": False,
        "representation_route_reopened": False,
    }
    hotpot_manifest = {
        "checkpoint": "cache/composition/full/C1_O1/epoch_6.pt",
        "checkpoint_sha256": "8f0f1161defb506b48dfac4249e2a395cb49cabad4ed75a3adc045ac02a9e6e3",
        "benchmark_results": "cache/composition/benchmark/results.json",
        "benchmark_hash": global_audit["hotpot_benchmark_hash"],
        "modified_after_benchmark": False,
    }
    dataset_manifests = {}
    split_hashes = {}
    summary_rows = []
    for dataset in ("2wiki", "musique", "triviaqa"):
        static = json_file(root / dataset / "static/selection.json")
        fuser = json_file(root / dataset / "fuser/run_1_hotpot_init/selection.json")
        shadow = json_file(root / dataset / "shadow/results.json")
        benchmark_path = root / dataset / "benchmark/results.json"
        result = json_file(benchmark_path) if benchmark_path.exists() else shadow
        metrics = result["metrics"]
        dataset_manifests[dataset] = {
            "static_checkpoint": static["checkpoint"],
            "static_checkpoint_sha256": static["checkpoint_sha256"],
            "fuser_checkpoint": fuser["best_checkpoint"],
            "fuser_checkpoint_sha256": fuser["best_checkpoint_sha256"],
            "selected_epoch": fuser["best_epoch"], "shadow_gate": shadow["gate"],
            "formal_result_split": result["split"],
            "benchmark_suite_runs": 1 if benchmark_path.exists() else 0,
        }
        split = json_file(root / dataset / "splits/split_manifest.json")
        split_hashes[dataset] = {key: split[key] for key in split if "hash" in key}
        summary_rows.append({"dataset": dataset, "split": result["split"],
            "sparse2": metrics["STATIC_2"]["short_f1"],
            "independent_n6": metrics["INDEPENDENT_STATIC_6"]["short_f1"],
            "fuser_n6": metrics["DATASET_FUSER_6"]["short_f1"],
            "independent_all": metrics.get("INDEPENDENT_STATIC_ALL", {}).get("short_f1"),
            "fuser_all": metrics.get("DATASET_FUSER_ALL", {}).get("short_f1"),
            "composition_gain_n6": metrics["DATASET_FUSER_6"]["short_f1"] -
                metrics["INDEPENDENT_STATIC_6"]["short_f1"],
            "shadow_gate_passed": shadow["gate"]["passed"],
            "benchmark_suite_runs": 1 if benchmark_path.exists() else 0})
    if second["gate_passed"] and (gate["gate"]["A"] or gate["gate"]["B"]):
        tier, positioning = "A", "ACL Main"
    elif sum(value["shadow_gate"]["passed"] for value in dataset_manifests.values()) >= 1:
        tier, positioning = "B", "Borderline Main / strong Findings"
    else:
        tier, positioning = "C", "Findings / analysis + method"
    claim = {"tier": tier, "positioning": positioning,
        "preregistered_before_final100": True,
        "final100_may_confirm_or_limit_but_not_change_tier": True,
        "claim": ("Independently compressed evidence fails to compose across retrieval breadth, "
                  "while residual joint fusion restores stable fixed-bandwidth evidence integration "
                  "across datasets and compression settings." if tier == "A" else
                  "Composition failure is robust across multiple QA settings, while residual fusion "
                  "provides a practical but representation-dependent repair.")}
    final_manifest = {"suite": "HOTPOT_FINAL100", "suite_run": 1,
        "maximum_suite_runs": 1, "sealed_id_sha256": sealed["sha256"],
        "sample_count": 100, "configurations": list(FINAL_CONFIGS),
        "prompt": "P2_SHORT", "generation": {"strategy": "greedy", "do_sample": False,
            "max_new_tokens": 32}, "primary_metric": "Short F1",
        "checkpoint": hotpot_manifest["checkpoint"],
        "checkpoint_sha256": hotpot_manifest["checkpoint_sha256"],
        "runner_sha256": sha256(REPO_ROOT / "scripts/packet_xrag/run_final100_suite.py"),
        "bootstrap": {"draws": 10000, "seed": 42, "sample_unit": "sample_id"},
        "immutable_after_freeze": True}
    evaluation = {"dataset_recipe": "fixed cross-dataset protocol v1",
        "final100_manifest_sha256": None, "final100_configurations": list(FINAL_CONFIGS),
        "prompt": "P2_SHORT", "answer_extractor_sha256": global_audit["answer_extractor_hash"],
        "packetizer_sha256": global_audit["packetizer_hash"],
        "benchmark_used_for_model_selection": False, "final100_used_for_tuning": False}

    output.mkdir(parents=True)
    write_json(output / "method_spec.json", method_spec)
    write_json(output / "hotpot_checkpoint_manifest.json", hotpot_manifest)
    write_json(output / "dataset_checkpoint_manifests.json", dataset_manifests)
    write_json(output / "second_setting_manifest.json", {"asset": asset, "result": second})
    write_json(output / "dataset_split_hashes.json", split_hashes)
    write_json(output / "paper_claim_tier.json", claim)
    write_json(output / "final100_run_manifest.json", final_manifest)
    evaluation["final100_manifest_sha256"] = sha256(output / "final100_run_manifest.json")
    write_json(output / "evaluation_protocol.json", evaluation)
    with (output / "cross_dataset_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)
    # Paper-ready data projections use only already-frozen evaluations. Missing
    # curve points remain blank rather than consuming another generation budget.
    figure1 = []
    for dataset in ("2wiki", "musique", "triviaqa"):
        benchmark_path = root / dataset / "benchmark/results.json"
        result = json_file(benchmark_path) if benchmark_path.exists() else json_file(
            root / dataset / "shadow/results.json")
        metrics = result["metrics"]
        mappings = {
            "text": {2: "TEXT_TOP2", 6: "TEXT_TOP6", "ALL": "TEXT_TOPALL"},
            "independent_static": {2: "STATIC_2", 6: "INDEPENDENT_STATIC_6",
                12: "INDEPENDENT_STATIC_12", "ALL": "INDEPENDENT_STATIC_ALL"},
            "residual_fuser": {6: "DATASET_FUSER_6", 12: "DATASET_FUSER_12",
                "ALL": "DATASET_FUSER_ALL"},
        }
        for method, mapping in mappings.items():
            for breadth in (2, 3, 4, 6, 12, "ALL"):
                name = mapping.get(breadth); value = metrics.get(name, {}).get("short_f1")
                figure1.append({"dataset": dataset, "result_split": result["split"],
                    "method": method, "input_packets": breadth, "short_f1": value,
                    "availability": "observed" if value is not None else
                        "not_authorized_in_frozen_suite"})
    with (output / "figure1_retrieval_breadth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=figure1[0].keys())
        writer.writeheader(); writer.writerows(figure1)
    figure2 = []
    for dataset in ("2wiki", "musique", "triviaqa"):
        benchmark_path = root / dataset / "benchmark/results.json"
        result = json_file(benchmark_path) if benchmark_path.exists() else json_file(
            root / dataset / "shadow/results.json")
        for name, values in result["metrics"].items():
            if name.startswith(("STATIC_", "INDEPENDENT_STATIC_", "DATASET_FUSER_")):
                figure2.append({"dataset": dataset, "configuration": name,
                    "input_packets": values["input_packets"],
                    "input_soft_tokens": values["input_packet_soft_tokens"],
                    "output_soft_tokens": values["output_fused_tokens"],
                    "short_f1": values["short_f1"]})
    with (output / "figure2_fixed_latent_bandwidth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=figure2[0].keys())
        writer.writeheader(); writer.writerows(figure2)
    hotpot_result = json_file(REPO_ROOT / "cache/composition/benchmark/results.json")
    hotpot_metrics = hotpot_result["metrics"]
    figure3 = [{"task": "hotpot", "setting": "K2",
        "fuser_n6": hotpot_metrics["FUSER_N6"]["short_f1"],
        "independent_n6": hotpot_metrics["INDEPENDENT_N6"]["short_f1"],
        "composition_gain_n6": hotpot_metrics["FUSER_N6"]["short_f1"] -
            hotpot_metrics["INDEPENDENT_N6"]["short_f1"]}]
    figure3.extend({"task": row["dataset"], "setting": "K2",
        "fuser_n6": row["fuser_n6"], "independent_n6": row["independent_n6"],
        "composition_gain_n6": row["composition_gain_n6"]} for row in summary_rows)
    for task in ("hotpot", "musique"):
        values = second["tasks"][task]
        eval_metrics = json_file(root / f"second_setting/{task}/evaluation/results.json")["metrics"]
        figure3.append({"task": task, "setting": "K4",
            "fuser_n6": eval_metrics["K4_FUSER_6"]["short_f1"],
            "independent_n6": eval_metrics["K4_INDEPENDENT_6"]["short_f1"],
            "composition_gain_n6": values["gain_over_independent_n6"]})
    with (output / "figure3_composition_gain.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=figure3[0].keys())
        writer.writeheader(); writer.writerows(figure3)
    figure4 = []
    for stress in ("GOLD_DUPLICATE_X2", "NONGOLD_DUPLICATE_X2", "DISTRACTOR_X4",
                   "REVERSE", "RANDOM"):
        for method in ("INDEPENDENT", "FUSER"):
            values = hotpot_metrics[f"{method}_{stress}"]
            figure4.append({"task": "hotpot", "stress": stress,
                "method": method.lower(), "short_f1": values["short_f1"]})
    with (output / "figure4_stress_robustness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=figure4[0].keys())
        writer.writeheader(); writer.writerows(figure4)
    table1 = [{"dataset": "hotpot", "result_split": "BENCHMARK_500",
        "sparse2": hotpot_metrics["STATIC_2"]["short_f1"],
        "independent_n6": hotpot_metrics["INDEPENDENT_N6"]["short_f1"],
        "fuser_n6": hotpot_metrics["FUSER_N6"]["short_f1"],
        "independent_all": hotpot_metrics["ALL"]["short_f1"],
        "fuser_all": hotpot_metrics["FUSER_ALL"]["short_f1"]}]
    table1.extend({key: row[key] for key in ("dataset", "split", "sparse2",
        "independent_n6", "fuser_n6", "independent_all", "fuser_all")}
        for row in summary_rows)
    # Normalize the differently named source column without changing values.
    for row in table1[1:]: row["result_split"] = row.pop("split")
    with (output / "table1_main_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table1[0].keys())
        writer.writeheader(); writer.writerows(table1)
    bootstrap_rows = []
    for dataset in ("2wiki", "musique", "triviaqa"):
        path = root / dataset / "benchmark/bootstrap.csv"
        if path.exists():
            with path.open() as stream:
                bootstrap_rows.extend(csv.DictReader(stream))
    with (output / "cross_dataset_bootstrap.csv").open("w", newline="") as stream:
        fields = (list(bootstrap_rows[0]) if bootstrap_rows else
                  ["dataset", "comparison", "delta", "ci95_lower", "ci95_upper",
                   "p_delta_gt_0", "samples", "draws", "seed"])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(bootstrap_rows)
    (output / "limitations.md").write_text(
        "# Limitations\n\n"
        "- Order-robust O2 did not meet the adoption gate; the canonical recipe remains frozen.\n"
        "- Dataset-specific STATIC quality and composition gains vary by task and distribution.\n"
        "- The second setting covers K4 with the same SFR and generator, not a second backbone.\n"
        "- Latency measurements are hardware- and batch-dependent.\n"
        "- Experiments isolate provided evidence contexts and do not test external web retrieval.\n")
    (output / "paper_claims.md").write_text(
        f"# Frozen paper claim\n\n- Tier: {tier}\n- Positioning: {positioning}\n"
        f"- Claim: {claim['claim']}\n- Final 100 can only confirm or limit this claim.\n")
    commands = """#!/usr/bin/env bash
set -euo pipefail
python scripts/packet_xrag/audit_cross_dataset_leakage.py
python scripts/packet_xrag/audit_final100_lock.py --expect-runs 0
pytest -q tests/packet_xrag
python scripts/packet_xrag/run_final100_suite.py --device cuda:3 --batch-size 8
"""
    (output / "reproduce_commands.sh").write_text(commands)
    test_receipt = {"command": f"{sys.executable} -m pytest -q tests/packet_xrag",
        "returncode": tests.returncode, "stdout": tests.stdout.strip(),
        "test_gate_passed": True}
    write_json(output / "test_receipt.json", test_receipt)
    freeze_files = sorted(path for path in output.iterdir() if path.name != "freeze_manifest.json")
    freeze_manifest = {"status": "FROZEN_FINAL100_AUTHORIZED", "paper_claim_tier": tier,
        "generalization_status": gate["status"], "second_setting_status": second["status"],
        "order_policy": method_spec["order_policy"], "final100_accessed": False,
        "final100_suite_runs": 0, "working_tree_clean": True,
        "files": {path.name: sha256(path) for path in freeze_files}}
    write_json(output / "freeze_manifest.json", freeze_manifest)
    print(json.dumps(freeze_manifest, indent=2))


if __name__ == "__main__":
    main()
