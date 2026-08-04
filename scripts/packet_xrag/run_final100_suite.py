#!/usr/bin/env python
"""Consume the sealed Hotpot final-100 exactly once with the frozen suite."""

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.bootstrap_cross_dataset_results import paired_bootstrap
from scripts.packet_xrag.build_controller_data import encode_samples
from scripts.packet_xrag.composition_training_common import (
    build_fuser, load_frozen_generator, load_frozen_k2_projector,
)
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.model import SFR
from src.packet_xrag.controller.feature_cache import (
    ControllerFeatureCache, ControllerFeatureWriter, make_candidate_packets,
    ordered_ids_sha256, sample_id, sha256_file,
)
from src.packet_xrag.generalization.dataset_evaluation import (
    evaluate_fuser, evaluate_independent, evaluate_no_context, make_c1_fused, summarize,
)
from src.packet_xrag.generalization.protocol import assert_manifest_immutable

CONFIGS = ("NO_CONTEXT", "TOPK_3", "STATIC_2", "INDEPENDENT_STATIC_6",
           "FUSER_N6", "FUSER_N12", "INDEPENDENT_ALL", "FUSER_ALL",
           "XRAG_ORACLE")
HOTPOT_FUSER_SHA256 = "8f0f1161defb506b48dfac4249e2a395cb49cabad4ed75a3adc045ac02a9e6e3"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_final_features(samples, final_dir, device, sample_batch_size=4):
    feature_dir = final_dir / "features"
    writer = ControllerFeatureWriter(feature_dir)
    tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    packet_count = support_count = 0
    for start in range(0, len(samples), sample_batch_size):
        entries = []
        for sample in samples[start:start + sample_batch_size]:
            packets, gold_ids = make_candidate_packets(sample)
            entries.append((sample, packets, gold_ids))
        embeddings = encode_samples(tokenizer, model, entries, device, 180)
        for (sample, packets, gold_ids), values in zip(entries, embeddings):
            writer.add(sample, packets, gold_ids, values)
            packet_count += len(packets); support_count += len(gold_ids)
        print(f"final100 features: {min(start + len(entries), len(samples))}/{len(samples)}",
              flush=True)
    manifest = writer.close({"split": "HOTPOT_FINAL100", "ordered_ids_sha256":
        ordered_ids_sha256([sample_id(sample) for sample in samples]),
        "effective_split_hash": ordered_ids_sha256([sample_id(sample) for sample in samples]),
        "quarantine_hash": "not_applicable_official_validation",
        "source_dataset_identifier": "hotpotqa/hotpot_qa:distractor:validation",
        "packet_construction_version": sha256_file(
            REPO_ROOT / "src/packet_xrag/controller/feature_cache.py"),
        "sfr_checkpoint_identifier": v1.SFR_MODEL_NAME,
        "query_encoding_template_hash": hashlib.sha256(
            "{question} (verbatim; no instruction)".encode()).hexdigest(),
        "number_of_samples": len(samples), "number_of_packets": packet_count,
        "gold_support_count": support_count,
        "creation_command": "single locked run_final100_suite.py process",
        "completion_status": "complete", "sfr_trainable_parameters": 0})
    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    root = Path(args.root); final = root / "final100"; frozen = root / "final_frozen"
    source_manifest = frozen / "final100_run_manifest.json"
    evaluation = json.loads((frozen / "evaluation_protocol.json").read_text())
    assert_manifest_immutable(source_manifest, evaluation["final100_manifest_sha256"])
    manifest = json.loads(source_manifest.read_text())
    if tuple(manifest["configurations"]) != CONFIGS or manifest["suite_run"] != 1:
        raise RuntimeError("frozen final-100 suite configuration mismatch")
    if sha256_file(REPO_ROOT / "scripts/packet_xrag/run_final100_suite.py") != manifest[
            "runner_sha256"]:
        raise RuntimeError("final-100 suite runner changed after freeze")
    freeze = json.loads((frozen / "freeze_manifest.json").read_text())
    if freeze["status"] != "FROZEN_FINAL100_AUTHORIZED":
        raise RuntimeError("unified freeze did not authorize final 100")
    for name, digest in freeze["files"].items():
        if sha256_file(frozen / name) != digest:
            raise RuntimeError(f"unified frozen file changed: {name}")
    if subprocess.run([sys.executable, "scripts/packet_xrag/audit_final100_lock.py",
                       "--expect-runs", "0"], cwd=REPO_ROOT).returncode:
        raise RuntimeError("final-100 pre-access lock audit failed")
    forbidden = [name for name in ("run_manifest.json", "predictions.jsonl", "summary.csv",
        "summary.md", "bootstrap.json", "final_decision.md", "final_report.json", "features")
        if (final / name).exists()]
    if forbidden:
        raise RuntimeError(f"final-100 output already materialized: {forbidden}")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    lock_path = final / "lock.json"; lock = json.loads(lock_path.read_text())
    sealed_path = final / "sealed_ids.json"; sealed = json.loads(sealed_path.read_text())
    if lock != {"split": "HOTPOT_FINAL100", "runs": 0, "maximum_runs": 1}:
        raise RuntimeError("final-100 one-suite lock is not pristine")
    if ledger["final100_accessed"] or ledger["final100_suite_runs"]:
        raise RuntimeError("final-100 ledger records prior access")
    if sealed["questions_contexts_answers_accessed"] or sealed["suite_runs"]:
        raise RuntimeError("sealed-ID manifest records prior content access")
    # Consume every access guard before loading validation content. Any failure
    # after this point is terminal and never authorizes a second suite.
    lock.update({"runs": 1, "checkpoint_sha256": HOTPOT_FUSER_SHA256,
                 "manifest_sha256": evaluation["final100_manifest_sha256"]})
    write_json(lock_path, lock)
    ledger["final100_accessed"] = True; ledger["final100_suite_runs"] = 1
    write_json(ledger_path, ledger)
    sealed["questions_contexts_answers_accessed"] = True; sealed["suite_runs"] = 1
    write_json(sealed_path, sealed)
    (final / "run_manifest.json").write_text(source_manifest.read_text())

    print("Final-100 lock consumed; loading sealed official validation rows", flush=True)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                           trust_remote_code=True)
    wanted = set(sealed["ordered_sample_ids"])
    by_id = {sample_id(sample): sample for sample in dataset
             if sample_id(sample) in wanted}
    if set(by_id) != set(sealed["ordered_sample_ids"]):
        raise RuntimeError("could not recover every sealed final-100 ID")
    samples = [by_id[sample_id_] for sample_id_ in sealed["ordered_sample_ids"]]
    if ordered_ids_sha256([sample_id(sample) for sample in samples]) != sealed["sha256"]:
        raise RuntimeError("materialized final-100 order/hash mismatch")
    feature_manifest = build_final_features(samples, final, args.device,
                                            args.feature_batch_size)
    cache = ControllerFeatureCache(final / "features")
    records = [cache[index] for index in range(len(cache))]
    device = torch.device(args.device); torch.cuda.set_device(device)
    scores = load_static_score_cache(cache,
        "cache/controller/static/best_short_f1/scorer.pt", final / "static_scores.pt", device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
        key=lambda index: (-float(scores[record["sample_id"]][index]), index))
        for record in records}
    tokenizer, generator, xrag_id, config = load_frozen_generator(device)
    k2 = load_frozen_k2_projector(config, device)
    checkpoint = Path(manifest["checkpoint"])
    if sha256_file(checkpoint) != HOTPOT_FUSER_SHA256:
        raise RuntimeError("frozen Hotpot fuser hash mismatch")
    fuser = build_fuser("C1").to(device)
    fuser.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True)["state_dict"], strict=True)
    fuser.eval(); rows_by_config = {}

    def add(name, rows):
        rows_by_config[name] = rows
        print(json.dumps({name: summarize(rows)}), flush=True)

    add("NO_CONTEXT", evaluate_no_context(records, tokenizer, generator, device,
                                            args.batch_size))
    groups = [record["topk_ranking"][:3] for record in records]
    add("TOPK_3", evaluate_independent("TOPK_3", records, groups, k2, tokenizer,
        generator, xrag_id, device, args.batch_size))
    groups = [rankings[record["sample_id"]][:2] for record in records]
    add("STATIC_2", evaluate_independent("STATIC_2", records, groups, k2, tokenizer,
        generator, xrag_id, device, args.batch_size))
    groups6 = [rankings[record["sample_id"]][:6] for record in records]
    add("INDEPENDENT_STATIC_6", evaluate_independent("INDEPENDENT_STATIC_6", records,
        groups6, k2, tokenizer, generator, xrag_id, device, args.batch_size))
    add("FUSER_N6", evaluate_fuser("FUSER_N6", fuser, records, groups6, make_c1_fused,
        k2, tokenizer, generator, xrag_id, device, args.batch_size))
    groups12 = [rankings[record["sample_id"]][:12] for record in records]
    add("FUSER_N12", evaluate_fuser("FUSER_N12", fuser, records, groups12, make_c1_fused,
        k2, tokenizer, generator, xrag_id, device, args.batch_size))
    all_groups = [rankings[record["sample_id"]] for record in records]
    add("INDEPENDENT_ALL", evaluate_independent("INDEPENDENT_ALL", records, all_groups,
        k2, tokenizer, generator, xrag_id, device, args.batch_size))
    add("FUSER_ALL", evaluate_fuser("FUSER_ALL", fuser, records, all_groups,
        make_c1_fused, k2, tokenizer, generator, xrag_id, device, args.batch_size))
    oracle = [list(record["gold_packet_ids"]) for record in records]
    add("XRAG_ORACLE", evaluate_independent("XRAG_ORACLE", records, oracle, k2,
        tokenizer, generator, xrag_id, device, args.batch_size))
    if tuple(rows_by_config) != CONFIGS:
        raise AssertionError("final-100 suite emitted a non-manifest configuration")
    metrics = {name: summarize(rows) for name, rows in rows_by_config.items()}
    with (final / "predictions.jsonl").open("w") as stream:
        for rows in rows_by_config.values():
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (final / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("configuration", "samples", "short_f1",
            "short_em", "empty", "input_packets", "input_packet_soft_tokens",
            "output_fused_tokens", "mean_total_latency_ms", "peak_vram_gb"))
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({key: value for key, value in
                {"configuration": name, **values}.items() if key in writer.fieldnames})
    bootstrap = {"draws": 10000, "seed": 42, "sample_unit": "sample_id",
        "comparisons": {
            "Fuser N6 - STATIC2": paired_bootstrap(rows_by_config["FUSER_N6"],
                                                    rows_by_config["STATIC_2"]),
            "Fuser N6 - Independent N6": paired_bootstrap(rows_by_config["FUSER_N6"],
                rows_by_config["INDEPENDENT_STATIC_6"]),
            "Fuser N12 - Independent N12": {"status": "NOT_ESTIMABLE",
                "reason": "INDEPENDENT_N12 was not authorized in the immutable final suite"},
            "Fuser ALL - Independent ALL": paired_bootstrap(rows_by_config["FUSER_ALL"],
                                                             rows_by_config["INDEPENDENT_ALL"]),
        }}
    write_json(final / "bootstrap.json", bootstrap)
    f6 = metrics["FUSER_N6"]["short_f1"]
    static2 = metrics["STATIC_2"]["short_f1"]
    independent6 = metrics["INDEPENDENT_STATIC_6"]["short_f1"]
    stable = metrics["FUSER_N12"]["short_f1"] >= f6 - 2 and metrics[
        "FUSER_ALL"]["short_f1"] >= f6 - 2
    if f6 > independent6 and f6 >= static2 - 1 and stable:
        decision = "A"
    elif f6 - independent6 >= 3:
        decision = "B"
    elif f6 <= independent6 and not stable:
        decision = "CONTRADICTION"
    else:
        decision = "MIXED_LIMITATION"
    report = {"status": "COMPLETE_FROZEN_NO_SELECTION", "final100_accessed": True,
        "final100_suite_runs": 1, "further_final100_runs_authorized": False,
        "sample_count": len(records), "sample_hash": sealed["sha256"],
        "metrics": metrics, "bootstrap": bootstrap, "confirmation": decision,
        "feature_manifest": feature_manifest, "checkpoint_sha256": HOTPOT_FUSER_SHA256,
        "manifest_sha256": evaluation["final100_manifest_sha256"],
        "hotpot_main_model_modified_after_benchmark": False,
        "controller_route_reopened": False, "representation_route_reopened": False}
    write_json(final / "final_report.json", report)
    lines = ["# Final 100 summary", "", "| Configuration | Short F1 | Short EM |",
             "|---|---:|---:|"]
    lines.extend(f"| {name} | {values['short_f1']:.4f} | {values['short_em']:.4f} |"
                 for name, values in metrics.items())
    (final / "summary.md").write_text("\n".join(lines) + "\n")
    (final / "final_decision.md").write_text(
        f"# Final decision\n\n- Confirmation: {decision}\n"
        "- Final 100 accessed: Yes\n- Final 100 suite runs: 1\n"
        "- Further final-100 access: Forbidden\n- No retraining or method changes authorized.\n")
    ledger = json.loads(ledger_path.read_text()); ledger["final100_status"] = "complete"
    ledger["final100_confirmation"] = decision; write_json(ledger_path, ledger)
    subprocess.run([sys.executable, "scripts/packet_xrag/audit_final100_lock.py",
                    "--expect-runs", "1"], cwd=REPO_ROOT, check=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
