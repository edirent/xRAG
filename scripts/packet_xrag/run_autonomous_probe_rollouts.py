#!/usr/bin/env python
"""Run true Stage-2 STATIC-prefix generation rollouts for promoted STOP branches."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag.build_autonomous_generator_state_cache import generate_features
from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.autonomous_features import (
    generator_numeric_features, retriever_state_features,
)
from src.packet_xrag.controller.autonomous_search import (
    SubsetFeatureCache, assert_only_search_dev, load_search_split, read_jsonl,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/internal_dev_static_scores.pt")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def rebuild_probe(serialized, input_dim, hidden):
    layers = ([torch.nn.Linear(input_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(.1),
               torch.nn.Linear(64, 1)] if hidden else [torch.nn.Linear(input_dim, 1)])
    model = torch.nn.Sequential(*layers)
    state = {key: torch.tensor(value) for key, value in serialized["state_dict"].items()}
    model.load_state_dict(state, strict=True); model.eval()
    center = torch.tensor(serialized["feature_center"]); scale = torch.tensor(serialized["feature_scale"])
    return model, center, scale


def stop_probability(model_bundle, values):
    model, center, scale = model_bundle
    inputs = (torch.tensor(values) - center) / scale
    with torch.inference_mode(): return float(model(inputs).squeeze().sigmoid())


def summary(rows):
    lengths = [row["num_packets"] for row in rows]
    result = {"samples": len(rows), "short_f1": 100 * mean(row["short_f1"] for row in rows),
            "short_em": 100 * mean(row["short_em"] for row in rows),
            "avg_packets": mean(lengths), "median_packets": median(lengths),
            "p90_packets": sorted(lengths)[int(.9 * (len(lengths) - 1))],
            "empty": sum(row["short_prediction"] == "[EMPTY]" for row in rows),
            "support_recall": mean(row["support_recall"] for row in rows),
            "full_support": mean(row["full_support"] for row in rows),
            "length_distribution": {str(k): lengths.count(k) for k in range(1, 5)}}
    if all("prompt_tokens" in row for row in rows):
        result["avg_prompt_tokens"] = mean(row["prompt_tokens"] for row in rows)
    if all("generated_tokens" in row for row in rows):
        result["avg_generated_tokens"] = mean(row["generated_tokens"] for row in rows)
    soft_key = ("total_soft_tokens" if all("total_soft_tokens" in row for row in rows)
                else "num_soft_tokens")
    if all(soft_key in row for row in rows):
        result["avg_soft_tokens"] = mean(row[soft_key] for row in rows)
    return result


def utility_audit(policies, grouped_labels):
    actions, stops = [], []
    for sid, policy in policies.items():
        for action in policy["actions"]:
            key = (sid, tuple(action["selected_packet_ids_before"]))
            matching = {row["candidate_packet_id"]: row for row in grouped_labels.get(key, [])}
            if action["packet_id"] in matching: actions.append(matching[action["packet_id"]])
        selected = tuple(policy["selected_packet_ids"])
        if (sid, selected) in grouped_labels:
            stops.append(max(row["delta_utility"] for row in grouped_labels[(sid, selected)]))
    harmful = sum(row["delta_utility"] < -.02 for row in actions)
    unnecessary = 0; auditable_continue_states = 0
    for sid, policy in policies.items():
        for action in policy["actions"]:
            key = (sid, tuple(action["selected_packet_ids_before"]))
            if key in grouped_labels:
                auditable_continue_states += 1
                unnecessary += max(row["delta_utility"] for row in grouped_labels[key]) <= .02
    return {"auditable_actions": len(actions),
            "harmful_addition_rate": harmful / len(actions) if actions else None,
            "auditable_stops": len(stops),
            "false_stop_rate": sum(value > .02 for value in stops) / len(stops) if stops else None,
            "auditable_continue_states": auditable_continue_states,
            "unnecessary_continuation_rate": unnecessary / auditable_continue_states
            if auditable_continue_states else None}


def fixed_static2_policy(record, ranking):
    selected = ranking[:2]
    return {"selected_packet_ids": selected,
            "actions": [{"packet_id": packet_id,
                         "selected_packet_ids_before": selected[:step]}
                        for step, packet_id in enumerate(selected)]}


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); root = Path(args.root); output_dir = root / "stage2"
    output = output_dir / "probe_rollout_results.json"
    if output.exists(): raise RuntimeError("refusing to overwrite Stage-2 rollouts")
    probes = json.loads((root / "stage1/feasibility_probe_results.json").read_text())
    if probes["promoted_branches"] != ["B5", "B1"]:
        raise RuntimeError("unexpected Stage-1 promotion set")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    if ledger["usage"]["search_dev_generation"] != 2:
        raise RuntimeError("unexpected generation budget before Stage 2")
    dev_split, shadow_split = load_search_split(root)
    subset_payload = json.loads((root / "splits/probe_subset_ids.json").read_text())
    subset_ids = subset_payload["ordered_sample_ids"]
    if set(subset_ids) & set(shadow_split["ordered_sample_ids"]):
        raise RuntimeError("shadow entered Stage-2 rollout")
    parent = ControllerFeatureCache(args.feature_cache)
    cache = SubsetFeatureCache(parent, subset_ids)
    assert_only_search_dev([record["sample_id"] for record in cache.records],
                           dev_split["ordered_sample_ids"], "Stage-2 cache")
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(parent, args.static_checkpoint,
                                             args.static_score_cache, device)
    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
                key=lambda index: (-float(static_scores[record["sample_id"]][index]), index))
                for record in cache.records}
    tokenizer, generator, xrag_id, hashes = initialize_generator(args.k2_training_config, device)
    prefix_rows = {}; generation_seconds = 0.0
    for budget in range(1, 5):
        states = []
        for index in range(len(cache)):
            record = cache[index]; sid = record["sample_id"]
            states.append({"sample_id": sid, "selected": rankings[sid][:budget],
                           "question": record["question"],
                           "embeddings": record["packet_embeddings"]})
        for start in range(0, len(states), args.batch_size):
            torch.cuda.synchronize(device); began = time.time()
            current = generate_features(tokenizer, generator, xrag_id,
                                        states[start:start + args.batch_size], device, 32)
            torch.cuda.synchronize(device); generation_seconds += time.time() - began
            for row in current:
                prefix_rows[(row["sample_id"], len(row["selected_packet_ids"]))] = row
        print(json.dumps({"stage2_prefix_budget": budget, "samples": len(states)}), flush=True)

    branch_models = {
        "B1": rebuild_probe(probes["branches"]["B1"]["model"], 8, False),
        "B5": rebuild_probe(probes["branches"]["B5"]["model"], 23, True),
    }
    rows_by_branch, policies_by_branch = {}, {}
    for branch_id in ("B1", "B5"):
        branch_rows, policies = [], {}
        for index in range(len(cache)):
            record = cache[index]; sid = record["sample_id"]; ranking = rankings[sid]
            selected, actions, probabilities = [], [], []
            for budget in range(1, 5):
                packet_id = ranking[budget - 1]
                actions.append({"packet_id": packet_id,
                                "selected_packet_ids_before": list(selected)})
                selected.append(packet_id); generated = prefix_rows[(sid, budget)]
                values = generator_numeric_features(generated)
                if branch_id == "B5":
                    values = retriever_state_features(record, selected, static_scores[sid]) + values
                probability = stop_probability(branch_models[branch_id], values)
                probabilities.append(probability)
                if probability >= .5: break
            final = prefix_rows[(sid, len(selected))]
            prediction = final["provisional_answer"]
            em, f1 = selector.score_prediction(prediction, record["answer"])
            gold = set(record["gold_packet_ids"]); selected_set = set(selected)
            branch_rows.append({"sample_id": sid, "configuration": branch_id,
                                "selected_packet_ids": selected,
                                "stop_probabilities": probabilities,
                                "short_prediction": prediction, "short_em": em, "short_f1": f1,
                                "num_packets": len(selected), "total_soft_tokens": 2 * len(selected),
                                "prompt_tokens": final["prompt_tokens"],
                                "generated_tokens": final["generated_length"],
                                "support_recall": len(gold & selected_set) / len(gold),
                                "full_support": float(gold.issubset(selected_set))})
            policies[sid] = {"selected_packet_ids": selected, "actions": actions}
        rows_by_branch[branch_id] = branch_rows; policies_by_branch[branch_id] = policies

    baseline_all = read_jsonl(root / "stage0/search_dev_baselines.jsonl")
    subset_set = set(subset_ids)
    baselines = {name: [row for row in baseline_all if row["configuration"] == name and
                        row["sample_id"] in subset_set] for name in ("TOPK_3", "STATIC_2")}
    baseline_summary = {name: summary(rows) for name, rows in baselines.items()}
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    grouped_labels = defaultdict(list)
    for row in labels.rows:
        if row["sample_id"] in subset_set:
            grouped_labels[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(row)
    static_policies = {cache[index]["sample_id"]: fixed_static2_policy(
                       cache[index], rankings[cache[index]["sample_id"]]) for index in range(len(cache))}
    baseline_audit = utility_audit(static_policies, grouped_labels)
    result_branches = {}
    measured_ms_per_forward = generation_seconds * 1000 / (len(cache) * 4)
    for branch_id in ("B1", "B5"):
        metrics = summary(rows_by_branch[branch_id]); audit = utility_audit(
            policies_by_branch[branch_id], grouped_labels)
        f1_delta = metrics["short_f1"] - baseline_summary["STATIC_2"]["short_f1"]
        packet_saving = 1 - metrics["avg_packets"] / 2
        harm_reduction = (1 - audit["harmful_addition_rate"] /
                          baseline_audit["harmful_addition_rate"]
                          if audit["harmful_addition_rate"] is not None and
                          baseline_audit["harmful_addition_rate"] else None)
        gates = {"f1_gain_ge_1.5": f1_delta >= 1.5,
                 "within_0.5_and_packet_save_ge_15pct": f1_delta >= -.5 and packet_saving >= .15,
                 "harm_save_ge_25pct_and_f1_drop_le_0.5": harm_reduction is not None and
                 harm_reduction >= .25 and f1_delta >= -.5}
        avg_forwards = metrics["avg_packets"]
        result_branches[branch_id] = {"metrics": metrics, "utility_audit": audit,
            "delta_vs_static2": {"short_f1": f1_delta, "packet_saving_fraction": packet_saving,
                                 "harm_reduction_fraction": harm_reduction},
            "cost": {"logical_generator_forwards_per_sample": avg_forwards,
                     "extra_forwards_vs_static2": avg_forwards - 1,
                     "measured_milliseconds_per_generator_forward": measured_ms_per_forward,
                     "estimated_policy_milliseconds_per_sample": measured_ms_per_forward * avg_forwards,
                     "relative_generator_cost_vs_static2": avg_forwards,
                     "maximum_selected_packets": 4, "candidate_probes_per_step": 0},
            "gates": gates, "passed": any(gates.values())}
    passed = [branch_id for branch_id in ("B1", "B5") if result_branches[branch_id]["passed"]]
    retained = sorted(passed, key=lambda branch_id: (
        -result_branches[branch_id]["metrics"]["short_f1"],
        result_branches[branch_id]["metrics"]["avg_packets"], branch_id))[:2]
    payload = {"status": "PASS" if retained else "MANDATORY_STOP_NO_STAGE2_BRANCH",
               "split": "SEARCH_DEV_PROBE150", "probe_hash": subset_payload["sha256"],
               "baselines": baseline_summary, "static2_utility_audit": baseline_audit,
               "branches": result_branches, "retained_branches": retained,
               "shared_prefix_generation_wall_seconds": generation_seconds,
               "checkpoint_hashes": hashes, "search_shadow_accessed": False,
               "benchmark_accessed": False, "final_100_accessed": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / "probe_rollout_predictions.jsonl").open("w") as stream:
        for branch_id in ("B1", "B5"):
            for row in rows_by_branch[branch_id]: stream.write(json.dumps(row) + "\n")
    (output_dir / "probe_rollout_results.md").write_text(
        "# Stage-2 Probe Rollouts\n\n" + f"- Status: {payload['status']}\n" +
        f"- Retained: {', '.join(retained) if retained else 'none'}\n" +
        "- SEARCH_SHADOW / benchmark / final 100 accessed: No\n")
    ledger["usage"]["search_dev_generation"] = 4
    ledger["stage2"] = {"status": payload["status"], "retained_branches": retained,
                        "probe_hash": subset_payload["sha256"], "branch_evaluations": 2}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "retained": retained,
                      "baseline": baseline_summary, "branches": result_branches}, indent=2), flush=True)


if __name__ == "__main__": main()
