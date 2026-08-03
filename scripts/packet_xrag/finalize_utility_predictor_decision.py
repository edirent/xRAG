#!/usr/bin/env python
"""Apply the deployment gate and create the final frozen package on PASS."""

import argparse
import csv
import hashlib
import inspect
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.token_resampler_common import sha256_file
from scripts.packet_xrag import run_selector_calibration as selector
from src.packet_xrag.controller.interaction_utility_predictor import INTERACTION_FEATURE_DIM
from src.packet_xrag.controller.state_shift_utility_predictor import STATE_FEATURE_DIM
from src.packet_xrag.controller.static_utility_predictor import BASE_FEATURE_DIM, PROJECTION_DIM
from src.packet_xrag.controller.utility_evaluation import summarize_generation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/utility_predictor")
    return parser.parse_args(argv)


def comparison(bootstrap, name):
    return next(row for row in bootstrap["comparisons"] if row["comparison"] == name)


def main(argv=None):
    args = parse_args(argv); root = Path(args.root); benchmark = root / "benchmark"
    b_gate = json.loads((root / "model_b/model_b_gate.json").read_text())
    c_gate = json.loads((root / "model_c/model_c_gate.json").read_text())
    if not b_gate["passed"]:
        raise RuntimeError("MANDATORY STOP: Model B gate did not pass")
    final_type = "C" if c_gate["passed"] else "B"
    manifest = json.loads((benchmark / "benchmark_manifest.json").read_text())
    if manifest["final_model_type"] != final_type:
        raise RuntimeError("benchmark final candidate differs from internal-dev gate")
    predictions = [json.loads(line) for line in (benchmark / "predictions.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in predictions: grouped[row["configuration"]].append(row)
    names = {"A": "MODEL_A_UTILITY_STOP", "B": "MODEL_B_STATE_SHIFT_STOP",
             "C": "MODEL_C_INTERACTION_STOP"}
    metrics = {name: summarize_generation(rows) for name, rows in grouped.items()}
    final_name = names[final_type]; final_metrics = metrics[final_name]
    bootstrap = json.loads((benchmark / "bootstrap_summary.json").read_text())
    utility = json.loads((benchmark / "policy_utility_summary.json").read_text())["configurations"]
    vs_static = comparison(bootstrap, "FINAL - STATIC_2")
    vs_topk = comparison(bootstrap, "FINAL - TOPK_3")
    vs_a = comparison(bootstrap, f"{final_type} - " + ("B" if final_type == "C" else "A"))
    b_vs_a = comparison(bootstrap, "B - A")
    static_harmful = utility["STATIC_2"]["harmful_addition_rate"]
    final_harmful = utility[final_name]["harmful_addition_rate"]
    harmful_reduction = ((static_harmful - final_harmful) / static_harmful
                         if static_harmful else None)
    packet_reduction = (2.0 - final_metrics["avg_packets"]) / 2.0
    harmful_improved = final_harmful < static_harmful
    model_b_stable = b_vs_a["short_f1_delta"] > 0 and b_vs_a["short_f1_ci95"][0] > 0
    gate_a = (vs_static["short_f1_delta"] >= 3 and vs_static["short_f1_ci95"][0] > 0
              and final_metrics["avg_packets"] <= 2.5 and vs_topk["short_f1_delta"] >= 6
              and vs_topk["short_f1_ci95"][0] > 0 and
              ((harmful_reduction is not None and harmful_reduction >= .25) or
               (harmful_reduction is None and final_harmful <= .15)))
    gate_b = (abs(vs_static["short_f1_delta"]) <= .5 and final_metrics["avg_packets"] <= 1.5
              and packet_reduction >= .25 and vs_static["short_f1_ci95"][1] >= 0)
    gate_c = (((vs_static["short_f1_delta"] >= 1.5 and vs_static["short_f1_ci95"][0] > 0) or
               (abs(vs_static["short_f1_delta"]) <= .5 and packet_reduction >= .15))
              and model_b_stable and harmful_improved)
    final_beats_a = metrics[final_name]["short_f1"] > metrics[names["A"]]["short_f1"]
    b_clearly_below_a = b_vs_a["short_f1_ci95"][1] < 0
    mandatory_failure = ((vs_static["short_f1_delta"] < 1.5 and packet_reduction < .15)
                         or not final_beats_a or b_clearly_below_a or not harmful_improved)
    if gate_a: selected_gate = "A"
    elif gate_b: selected_gate = "B"
    elif gate_c: selected_gate = "C"
    else: selected_gate = "FAIL"
    if mandatory_failure: selected_gate = "FAIL"
    conclusion = ({"A": "Strong deployment feasibility pass.",
                   "B": "Efficiency deployment feasibility pass.",
                   "C": "Utility prediction is viable but natural-data gains are moderate. Proceed only to controlled-redundancy validation.",
                   "FAIL": "Generator-defined state-dependent utility is strong as an oracle, but the deployable utility predictor does not capture enough signal to justify a controller method."}[selected_gate])
    selected_model_dir = root / f"model_{final_type.lower()}"
    frozen_selection = json.loads((selected_model_dir / "frozen_selection.json").read_text())
    payload = {"final_model_type": final_type, "final_configuration": final_name,
               "deployment_gate": selected_gate, "scientific_conclusion": conclusion,
               "benchmark_metrics": final_metrics, "delta_vs_static": vs_static,
               "delta_vs_topk": vs_topk, "model_b_vs_a": b_vs_a,
               "harmful_addition_rate": final_harmful,
               "static_harmful_addition_rate": static_harmful,
               "harmful_addition_reduction": harmful_reduction,
               "packet_reduction_vs_static2": packet_reduction,
               "model_b_stable_over_a": model_b_stable, "final_beats_model_a": final_beats_a,
               "controlled_redundancy_authorized": selected_gate in {"A", "B", "C"},
               "final_model_frozen": selected_gate in {"A", "B", "C"},
               "ready_for_final_100": False, "final_100_accessed": False,
               "final_100_runs": 0}
    (root / "final_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if selected_gate != "FAIL":
        frozen_dir = root / "final_frozen"; frozen_dir.mkdir(parents=True, exist_ok=True)
        source_checkpoint = Path(frozen_selection["checkpoint"])
        shutil.copyfile(source_checkpoint, frozen_dir / "final_model.pt")
        checkpoint_hash = sha256_file(frozen_dir / "final_model.pt")
        training_config = json.loads((selected_model_dir / "training_config.json").read_text())
        architecture = {
            "input_embedding_dim": 4096,
            "projection_dim": PROJECTION_DIM,
            "base_mlp_dimensions": [BASE_FEATURE_DIM, 2048, 512, 1],
            "state_shift_mlp_dimensions": [STATE_FEATURE_DIM, 1024, 256, 1],
            "interaction_mlp_dimensions": [INTERACTION_FEATURE_DIM, 1024, 256, 1],
            "base_features": ["projected_query", "projected_packet", "elementwise_product",
                              "absolute_difference", "projected_cosine", "original_sfr_cosine",
                              "static_relevance", "normalized_sentence_position",
                              "normalized_document_length"],
            "state_features": ["selected_mean", "selected_max", "query_times_selected_mean",
                               "query_abs_selected_mean", "selected_count_over_6",
                               "selected_static_mean", "selected_static_max",
                               "selected_query_cosine_mean", "selected_query_cosine_max",
                               "unique_document_count_over_6", "same_document_pair_fraction"],
            "interaction_features": ["projected_candidate", "selected_mean", "selected_max",
                                     "candidate_times_mean", "candidate_abs_mean",
                                     "candidate_times_max", "candidate_abs_max",
                                     "candidate_selected_cosine_max_mean_min",
                                     "candidate_document_selected", "same_document_count_over_6",
                                     "candidate_adjacent_to_selected"],
        }
        answer_extractor_hash = hashlib.sha256(
            inspect.getsource(selector.extract_short_answer).encode()
        ).hexdigest()
        (frozen_dir / "final_model_config.json").write_text(json.dumps({
            **training_config, "selected_epoch": frozen_selection["epoch"],
            "tau": frozen_selection["tau"], "deployment_gate": selected_gate,
            "architecture": architecture,
            "loss_weights": {"huber": 1.0, "pairwise": 0.5, "sign": 0.25},
            "huber_delta": 0.1, "pair_min_raw_gap": 0.05,
            "max_pairs_per_state": 16, "sign_near_zero_threshold": 0.02,
        }, indent=2, sort_keys=True) + "\n")
        checkpoint_audit = json.loads((root / "checkpoint_audit.json").read_text())
        (frozen_dir / "checkpoint_manifest.json").write_text(json.dumps({
            "model_type": final_type, "checkpoint_sha256": checkpoint_hash,
            "train_hash": checkpoint_audit["effective_train_hash"],
            "dev_hash": checkpoint_audit["internal_dev_hash"],
            "benchmark_hash": checkpoint_audit["benchmark_hash"],
            "prompt_hash": checkpoint_audit["prompt_hash"],
            "answer_mask_hash": checkpoint_audit["answer_mask_implementation_hash"],
            "answer_extractor_hash": answer_extractor_hash,
            "representation_hashes": {"v1": checkpoint_audit["v1_sha256"],
                                      "k2": checkpoint_audit["k2_sha256"]},
        }, indent=2, sort_keys=True) + "\n")
        (frozen_dir / "checkpoint_hashes.json").write_text(json.dumps({
            "final_model": checkpoint_hash, "v1": checkpoint_audit["v1_sha256"],
            "k2": checkpoint_audit["k2_sha256"], "static": checkpoint_audit["static_sha256"],
        }, indent=2, sort_keys=True) + "\n")
        shutil.copyfile(root / "utility_target_stats.json", frozen_dir / "utility_target_stats.json")
        shutil.copyfile(selected_model_dir / "frozen_selection.json", frozen_dir / "internal_dev_selection.json")
        shutil.copyfile(benchmark / "summary.csv", frozen_dir / "benchmark_summary.csv")
        shutil.copyfile(benchmark / "bootstrap_summary.csv", frozen_dir / "bootstrap_summary.csv")
        mechanism_lines = [
            "# Mechanism Summary", "",
            f"- Static utility gain: {comparison(bootstrap, 'A - STATIC_2')['short_f1_delta']:.4f} F1",
            f"- State sufficiency gain: {b_vs_a['short_f1_delta']:.4f} F1",
            f"- Interaction gain: {comparison(bootstrap, 'C - B')['short_f1_delta']:.4f} F1",
        ]
        for model_type, name in names.items():
            current_utility = utility[name]
            current_metrics = metrics[name]
            mechanism_lines.append(
                f"- Model {model_type}: harmful {current_utility['harmful_addition_rate']:.6f}; "
                f"missed-positive STOP {current_utility['missed_positive_stop_rate']}; "
                f"avg packets {current_metrics['avg_packets']:.4f}; "
                f"lengths {json.dumps(current_metrics['length_distribution'], sort_keys=True)}"
            )
        (frozen_dir / "mechanism_summary.md").write_text("\n".join(mechanism_lines) + "\n")
        (frozen_dir / "final_decision.md").write_text(
            f"# Final Utility Controller Decision\n\n- Final model: Model {final_type}\n"
            f"- Deployment Gate: {selected_gate}\n- Benchmark Short F1: {final_metrics['short_f1']:.4f}\n"
            f"- Avg packets: {final_metrics['avg_packets']:.4f}\n- Final 100 accessed: No\n"
            "- Final 100 runs: 0\n- Ready for final 100: No\n"
        )
        (frozen_dir / "reproduce_commands.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"# Frozen Model {final_type}; tau={frozen_selection['tau']}\n"
            "PYTHONPATH=. python scripts/packet_xrag/run_utility_predictor_benchmark.py "
            f"--final-model-type {final_type}\n"
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
