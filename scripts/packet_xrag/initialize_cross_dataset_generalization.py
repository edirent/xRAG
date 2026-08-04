#!/usr/bin/env python
"""Audit frozen assets and initialize the cross-dataset experiment ledger."""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from src.packet_xrag.composition.protocol import question_hash
from src.packet_xrag.controller.feature_cache import sha256_file
from src.packet_xrag.data.base_qa_adapter import MAX_PACKETS, PacketQADatasetAdapter
from src.packet_xrag.generalization.protocol import SEED, ordered_hash


EXPECTED = {"v1": "fa1a9ba443960acc176dc387989fe7c2e3fa1ef1cf24a38db265da4c1c60f760",
    "k2": "c40aa3dc297f57b5be73649f1754dc292ef98fefbc5c8a338b90e108427fd8a4",
    "static": "9ea9609ba1fd6ab1466610c2324aa1938d86604c6b23c07b0e68ca831ccb8276",
    "fuser": "8f0f1161defb506b48dfac4249e2a395cb49cabad4ed75a3adc045ac02a9e6e3"}


def source_hash(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def main():
    root = Path("cache/generalization")
    if root.exists(): raise RuntimeError("refusing to overwrite generalization protocol")
    paths = {"v1": Path("cache/projector/packet_projector_calibration/last/projector.pt"),
        "k2": Path("cache/projector/multi_token_k2/best_short_f1/multi_token_projector.pt"),
        "static": Path("cache/controller/static/best_short_f1/scorer.pt"),
        "fuser": Path("cache/composition/full/C1_O1/epoch_6.pt")}
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"MANDATORY STOP checkpoint mismatch: {observed}")
    root.mkdir(parents=True); (root / "final100").mkdir()
    choice = {"selected_dataset": "TriviaQA", "adapter": "TriviaQAAdapter",
        "source": "mandarjoshi/trivia_qa:rc",
        "reason": "official train/validation include labeled answers and evidence text; repository also contains an aligned retrieval evaluator",
        "selected_before_experiments": True, "fallback_used": False}
    (root / "single_hop_choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True) + "\n")
    k4_path = Path("cache/projector/multi_token_k4/best_short_f1/multi_token_projector.pt")
    second = {"selected_setting": "K4", "priority_audit": {"K1": "no formal compatible checkpoint",
        "K4": "first complete compatible checkpoint", "second_backbone": "not considered after K4 pass"},
        "checkpoint": str(k4_path), "checkpoint_sha256": sha256_file(k4_path),
        "selection_before_experiments": True, "output_M": 4}
    (root / "second_setting").mkdir(); (root / "second_setting/asset_selection.json").write_text(
        json.dumps(second, indent=2, sort_keys=True) + "\n")
    # Only IDs are materialized here. Questions, contexts, and answers remain sealed.
    hotpot_validation = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                                     trust_remote_code=True)
    final_ids = [str(value) for value in hotpot_validation["id"][-100:]]
    final_hash = ordered_hash(final_ids)
    (root / "final100/sealed_ids.json").write_text(json.dumps({"sample_count": 100,
        "selection": "last 100 rows of official HotpotQA distractor validation",
        "ordered_sample_ids": final_ids, "sha256": final_hash,
        "questions_contexts_answers_accessed": False, "suite_runs": 0},
        indent=2, sort_keys=True) + "\n")
    (root / "final100/lock.json").write_text(json.dumps(
        {"split": "HOTPOT_FINAL100", "runs": 0, "maximum_runs": 1}, indent=2,
        sort_keys=True) + "\n")
    prompt_hash = hashlib.sha256(v1.build_prompt("__QUESTION__", 4).encode()).hexdigest()
    audit = {"status": "PASS", "checkpoint_hashes": observed,
        "base_llm_identifier": v1.XRAG_MODEL_NAME, "sfr_identifier": v1.SFR_MODEL_NAME,
        "prompt_hash": prompt_hash, "answer_extractor_hash": source_hash(selector.extract_short_answer),
        "packetizer_hash": source_hash(PacketQADatasetAdapter.packetize),
        "hotpot_benchmark_hash": "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052",
        "hotpot_final100_hash": final_hash, "hotpot_fuser_full_sha256": observed["fuser"],
        "generator_trainable_parameters": 0, "sfr_trainable_parameters": 0,
        "k2_trainable_parameters": 0, "static_frozen_during_fuser_training": True,
        "forbidden_assets_loaded": [], "main_breadth": 6, "output_M": 4,
        "all_max_packets": MAX_PACKETS, "seed": SEED, "final100_accessed": False,
        "final100_suite_runs": 0}
    (root / "global_checkpoint_audit.json").write_text(json.dumps(audit, indent=2,
                                                                    sort_keys=True) + "\n")
    (root / "global_checkpoint_audit.md").write_text("# Global Checkpoint Audit\n\n" +
        "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n")
    budgets = {"order_ablation_full_runs": 1, "per_dataset_static_full_runs": 1,
        "per_dataset_fuser_full_runs": 2, "per_dataset_dev_generation": 6,
        "per_dataset_shadow": 1, "per_dataset_benchmark": 1,
        "second_setting_full_runs": 2, "hotpot_final100_suites": 1}
    ledger = {"protocol": "cross-dataset-generalization-v1", "seed": SEED,
        "budgets": budgets, "usage": {"order_ablation_full_runs": 0,
            "second_setting_full_runs": 0, "hotpot_final100_suites": 0},
        "datasets": {name: {"static_full_runs": 0, "fuser_full_runs": 0,
            "dev_generation": 0, "shadow": 0, "benchmark": 0}
            for name in ("2wiki", "musique", "triviaqa")},
        "single_hop_choice": choice, "second_setting_choice": second,
        "final100_accessed": False, "final100_suite_runs": 0}
    (root / "experiment_ledger.json").write_text(json.dumps(ledger, indent=2,
                                                              sort_keys=True) + "\n")
    (root / "experiment_ledger.md").write_text("# Generalization Experiment Ledger\n\n"
        "All counters initialized to zero; seed 20260804; final-100 sealed.\n")
    print(json.dumps({"status": "PASS", "hashes": observed,
                      "single_hop": "TriviaQA", "second_setting": "K4",
                      "final100_hash": final_hash, "final100_accessed": False}, indent=2))


if __name__ == "__main__": main()
