#!/usr/bin/env python
"""Run the locked eight Stage-1 feasibility probes and promote at most three branches."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.autonomous_features import (
    candidate_metrics, candidate_retriever_features, candidate_text_features,
    fit_torch_probe, generator_numeric_features, overlap, retriever_state_features,
    stop_metrics,
)
from src.packet_xrag.controller.autonomous_search import (
    SubsetFeatureCache, assert_only_search_dev, load_search_split, read_jsonl,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.static_utility_predictor import StaticUtilityPredictor
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset
from src.packet_xrag.controller.utility_training import UtilityFeatureStore


SEED = 20260804


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/internal_dev_static_scores.pt")
    parser.add_argument("--model-a-checkpoint", default="cache/controller/utility_predictor/model_a/epoch_4/model.pt")
    parser.add_argument("--bugfix-rerun", action="store_true",
                        help="Recompute the same probes after a metrics-only implementation fix.")
    return parser.parse_args(argv)


def fit_candidate(train_x, train_rows, eval_x, eval_rows, hidden):
    train_y = [max(-4.0, min(4.0, float(row["delta_utility"]))) / 4.0
               for row in train_rows]
    predictions, model = fit_torch_probe(train_x, train_y, eval_x, hidden=hidden,
                                         classification=False, epochs=350)
    predictions = [value * 4.0 for value in predictions]
    return candidate_metrics(predictions, eval_rows), predictions, model


def state_text(record, selected, candidate):
    selected_text = " ".join(record["packets"][index]["encoder_text"] for index in selected)
    candidate_text = record["packets"][candidate]["encoder_text"]
    return f"question {record['question']} selected {selected_text} candidate {candidate_text}"


def main(argv=None):
    args = parse_args(argv); root = Path(args.root); output_dir = root / "stage1"
    output = output_dir / "feasibility_probe_results.json"
    if output.exists() and not args.bugfix_rerun:
        raise RuntimeError("refusing to overwrite Stage-1 probe results")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    expected_usage = 8 if args.bugfix_rerun else 0
    if ledger["usage"]["feasibility_probes"] != expected_usage:
        raise RuntimeError("Stage-1 probe budget was already consumed")
    dev_split, shadow_split = load_search_split(root)
    probe_payload = json.loads((root / "splits/probe_subset_ids.json").read_text())
    eval_ids = set(probe_payload["ordered_sample_ids"])
    all_ids = set(dev_split["ordered_sample_ids"]); train_ids = all_ids - eval_ids
    if len(train_ids) != 200 or len(eval_ids) != 150 or eval_ids & set(shadow_split["ordered_sample_ids"]):
        raise RuntimeError("locked Stage-1 discovery/probe split invalid")
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    rows = [row for row in labels.rows if row["sample_id"] in all_ids]
    assert_only_search_dev([row["sample_id"] for row in rows], all_ids, "Stage-1 labels")
    grouped = defaultdict(list)
    for row in rows: grouped[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(row)
    generator_rows = read_jsonl(output_dir / "generator_state_features.jsonl")
    generator = {(row["sample_id"], tuple(row["selected_packet_ids"])): row
                 for row in generator_rows}
    intervention_rows = read_jsonl(output_dir / "candidate_intervention_features.jsonl")
    intervention = {(row["sample_id"], tuple(row["selected_packet_ids"]),
                     row["candidate_packet_id"]): row for row in intervention_rows}
    if set(grouped) != set(generator): raise RuntimeError("generator feature/state mismatch")
    parent = ControllerFeatureCache(args.feature_cache)
    cache = SubsetFeatureCache(parent, dev_split["ordered_sample_ids"])
    records = {cache[index]["sample_id"]: cache[index] for index in range(len(cache))}
    static_scores = load_static_score_cache(parent, args.static_checkpoint,
                                             args.static_score_cache, torch.device("cpu"))

    state_train, state_eval = [], []
    for key, state_rows in sorted(grouped.items()):
        sid, selected = key; record = records[sid]
        base = retriever_state_features(record, selected, static_scores[sid])
        gen = generator_numeric_features(generator[key])
        answer_rel = [overlap(generator[key]["provisional_answer"], record["question"]),
                      len(generator[key]["provisional_answer"].split()) / 16.0]
        item = {"key": key, "label": max(row["delta_utility"] for row in state_rows) <= .02,
                "retriever": base, "generator": gen,
                "combined": base + gen + answer_rel}
        (state_train if sid in train_ids else state_eval).append(item)
    train_y = [item["label"] for item in state_train]; eval_y = [item["label"] for item in state_eval]

    # Probe 1 and 2: mandatory retriever-only linear and two-layer MLP baselines.
    retriever_results = {}
    for name, hidden in (("P0_RETRIEVER_LINEAR", False), ("P1_RETRIEVER_MLP", True)):
        probabilities, model = fit_torch_probe(
            [item["retriever"] for item in state_train], train_y,
            [item["retriever"] for item in state_eval], hidden=hidden)
        retriever_results[name] = {"stop": stop_metrics(probabilities, eval_y),
                                   "model": model}

    train_candidate_rows = [row for row in rows if row["sample_id"] in train_ids]
    eval_candidate_rows = [row for row in rows if row["sample_id"] in eval_ids]
    def retriever_candidate_x(items):
        return [candidate_retriever_features(records[row["sample_id"]],
                    row["candidate_packet_id"], row["selected_packet_ids"],
                    static_scores[row["sample_id"]]) for row in items]
    for name, hidden in (("P0_RETRIEVER_LINEAR", False), ("P1_RETRIEVER_MLP", True)):
        metrics, _, model = fit_candidate(
            retriever_candidate_x(train_candidate_rows), train_candidate_rows,
            retriever_candidate_x(eval_candidate_rows), eval_candidate_rows, hidden)
        retriever_results[name]["candidate"] = metrics
        retriever_results[name]["candidate_model"] = model

    # Fair fixed-subset Model-A reference.
    model_a = StaticUtilityPredictor()
    model_a.load_state_dict(torch.load(args.model_a_checkpoint, map_location="cpu",
                                       weights_only=True), strict=True); model_a.eval()
    store = UtilityFeatureStore(parent, static_scores); model_a_predictions = []
    with torch.inference_mode():
        for start in range(0, len(eval_candidate_rows), 512):
            batch_rows = eval_candidate_rows[start:start + 512]
            batch = store.make_batch(batch_rows, torch.device("cpu"))
            model_a_predictions.extend(float(value) * 4.0 for value in model_a(**batch))
    model_a_metrics = candidate_metrics(model_a_predictions, eval_candidate_rows)

    branches = {}
    # Probe 3, B1: direct STOP from generator token statistics.
    probs, model = fit_torch_probe([item["generator"] for item in state_train], train_y,
                                   [item["generator"] for item in state_eval], hidden=False)
    branches["B1"] = {"metrics": {"stop": stop_metrics(probs, eval_y)}, "model": model}

    # Probe 4, B2: answer-conditioned lexical candidate relations.
    def answer_x(items):
        output_values = []
        for row in items:
            sid = row["sample_id"]; key = (sid, tuple(row["selected_packet_ids"]))
            output_values.append(candidate_retriever_features(
                records[sid], row["candidate_packet_id"], row["selected_packet_ids"],
                static_scores[sid]) + candidate_text_features(
                records[sid], row["candidate_packet_id"], row["selected_packet_ids"],
                generator[key]["provisional_answer"], static_scores[sid]) +
                generator_numeric_features(generator[key]))
        return output_values
    metrics, _, model = fit_candidate(
        answer_x(train_candidate_rows), train_candidate_rows,
        answer_x(eval_candidate_rows), eval_candidate_rows, True)
    branches["B2"] = {"metrics": {"candidate": metrics}, "model": model}

    # Probe 5, B3: candidate intervention self-likelihood shift.
    train_intervention_rows = [row for row in train_candidate_rows if
        (row["sample_id"], tuple(row["selected_packet_ids"]), row["candidate_packet_id"]) in intervention]
    eval_intervention_rows = [row for row in eval_candidate_rows if
        (row["sample_id"], tuple(row["selected_packet_ids"]), row["candidate_packet_id"]) in intervention]
    prediction_by_key = {
        (row["sample_id"], tuple(row["selected_packet_ids"]), row["candidate_packet_id"]): prediction
        for row, prediction in zip(eval_candidate_rows, model_a_predictions)
    }
    model_a_intervention_metrics = candidate_metrics([
        prediction_by_key[(row["sample_id"], tuple(row["selected_packet_ids"]),
                           row["candidate_packet_id"])]
        for row in eval_intervention_rows
    ], eval_intervention_rows)
    def intervention_x(items, include_answer=True):
        values = []
        for row in items:
            sid = row["sample_id"]; selected = tuple(row["selected_packet_ids"])
            feature = intervention[(sid, selected, row["candidate_packet_id"])]
            base = candidate_retriever_features(records[sid], row["candidate_packet_id"],
                                                selected, static_scores[sid])
            if include_answer:
                base += candidate_text_features(records[sid], row["candidate_packet_id"],
                                                selected, generator[(sid, selected)]["provisional_answer"],
                                                static_scores[sid])
            values.append(base + generator_numeric_features(generator[(sid, selected)]) +
                          [feature["base_self_nll"], feature["candidate_self_nll"],
                           feature["self_likelihood_shift"]])
        return values
    metrics, _, model = fit_candidate(
        intervention_x(train_intervention_rows, False), train_intervention_rows,
        intervention_x(eval_intervention_rows, False), eval_intervention_rows, True)
    branches["B3"] = {"metrics": {"candidate": metrics}, "model": model}

    # Probe 6, B4: deployable sparse text relation model.
    train_text = [state_text(records[row["sample_id"]], row["selected_packet_ids"],
                             row["candidate_packet_id"]) for row in train_candidate_rows]
    eval_text = [state_text(records[row["sample_id"]], row["selected_packet_ids"],
                            row["candidate_packet_id"]) for row in eval_candidate_rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                 sublinear_tf=True)
    train_sparse = vectorizer.fit_transform(train_text); eval_sparse = vectorizer.transform(eval_text)
    text_model = Ridge(alpha=10.0, solver="lsqr").fit(
        train_sparse, [max(-4.0, min(4.0, row["delta_utility"])) for row in train_candidate_rows])
    b4_predictions = text_model.predict(eval_sparse).tolist()
    branches["B4"] = {"metrics": {"candidate": candidate_metrics(
        b4_predictions, eval_candidate_rows)},
        "model": {"vocabulary_size": len(vectorizer.vocabulary_), "alpha": 10.0}}

    # Probe 7, B5: generator + STATIC-prefix STOP MLP.
    probs, model = fit_torch_probe([item["retriever"] + item["generator"] for item in state_train],
                                   train_y, [item["retriever"] + item["generator"]
                                                  for item in state_eval], hidden=True)
    branches["B5"] = {"metrics": {"stop": stop_metrics(probs, eval_y)}, "model": model}

    # Probe 8, B6: calibrated hybrid of answer, generator, retriever, and intervention signals.
    metrics, _, model = fit_candidate(
        intervention_x(train_intervention_rows, True), train_intervention_rows,
        intervention_x(eval_intervention_rows, True), eval_intervention_rows, True)
    branches["B6"] = {"metrics": {"candidate": metrics}, "model": model}

    baseline_stop = retriever_results["P1_RETRIEVER_MLP"]["stop"]
    baseline_candidate = model_a_metrics
    for branch_id, branch_result in branches.items():
        gates = {}
        if "stop" in branch_result["metrics"]:
            metric = branch_result["metrics"]["stop"]
            gates = {"stop_auroc_ge_0.70": metric["auroc"] >= .70,
                     "stop_auprc_gain_ge_0.10": metric["auprc"] - baseline_stop["auprc"] >= .10}
        if "candidate" in branch_result["metrics"]:
            metric = branch_result["metrics"]["candidate"]
            reference = (model_a_intervention_metrics if branch_id in {"B3", "B6"}
                         else baseline_candidate)
            gates = {"within_state_spearman_ge_0.20": metric["within_state_spearman"] >= .20,
                     "top1_gain_vs_model_a_ge_0.08": metric["best_action_top1_accuracy"] -
                     reference["best_action_top1_accuracy"] >= .08,
                     "regret_reduction_vs_model_a_ge_25pct": metric["teacher_policy_regret"] <=
                     .75 * reference["teacher_policy_regret"]}
            branch_result["model_a_same_pool_reference"] = reference
        branch_result["gates"] = gates; branch_result["passed_any_gate"] = any(gates.values())
    passed = [branch_id for branch_id, value in branches.items() if value["passed_any_gate"]]
    def promotion_score(branch_id):
        value = branches[branch_id]["metrics"]
        if "stop" in value:
            return max(value["stop"]["auroc"] - .70,
                       value["stop"]["auprc"] - baseline_stop["auprc"] - .10)
        candidate = value["candidate"]
        reference = branches[branch_id]["model_a_same_pool_reference"]
        return max(candidate["within_state_spearman"] - .20,
                   candidate["best_action_top1_accuracy"] - reference["best_action_top1_accuracy"] - .08,
                   .75 * reference["teacher_policy_regret"] - candidate["teacher_policy_regret"])
    promoted = sorted(passed, key=lambda branch_id: (-promotion_score(branch_id), branch_id))[:3]
    payload = {"status": "PASS" if promoted else "MANDATORY_STOP_ALL_PROBES_FAILED",
               "seed": SEED, "split": {"discovery_samples": 200, "probe_samples": 150,
               "probe_hash": probe_payload["sha256"]}, "probe_count": 8,
               "retriever_baselines": retriever_results, "model_a_reference": model_a_metrics,
               "model_a_intervention_pool_reference": model_a_intervention_metrics,
               "branches": branches, "promoted_branches": promoted,
               "promotion_cap": 3, "search_shadow_accessed": False,
               "benchmark_accessed": False, "final_100_accessed": False}
    payload["metrics_bugfix_rerun"] = bool(args.bugfix_rerun)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "feasibility_probe_results.md").write_text(
        "# Stage-1 Feasibility Probes\n\n" +
        f"- Status: {payload['status']}\n- Promoted: {', '.join(promoted) if promoted else 'none'}\n"
        f"- SEARCH_SHADOW accessed: No\n- Benchmark accessed: No\n- Final 100 accessed: No\n")
    ledger["usage"]["feasibility_probes"] = 8
    ledger["stage1"] = {"status": payload["status"], "promoted_branches": promoted,
                        "probe_count": 8, "probe_hash": probe_payload["sha256"]}
    for entry in ledger["branches"]:
        result = branches[entry["experiment_id"]]
        entry.update({"status": "completed", "actual_metrics": result["metrics"],
                      "result": "promoted" if entry["experiment_id"] in promoted else "rejected",
                      "decision": "keep" if entry["experiment_id"] in promoted else "reject",
                      "reason": "passed a preregistered Stage-1 gate" if result["passed_any_gate"] else
                                "failed all preregistered Stage-1 gates"})
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "promoted": promoted,
                      "model_a": model_a_metrics,
                      "branch_metrics": {key: value["metrics"] for key, value in branches.items()}},
                     indent=2), flush=True)


if __name__ == "__main__": main()
