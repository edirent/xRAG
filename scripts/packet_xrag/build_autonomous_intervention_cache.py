#!/usr/bin/env python
"""Measure deployable provisional-answer self-likelihood under candidate additions."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.packet_xrag.controller.autonomous_features import sanitize_provisional_answer
from src.packet_xrag.controller.autonomous_search import (
    SubsetFeatureCache, assert_no_inference_leakage, assert_only_search_dev,
    load_search_split, read_jsonl,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.generator_utility import gold_answer_nll_batch
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
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv); root = Path(args.root); output_dir = root / "stage1"
    output = output_dir / "candidate_intervention_features.jsonl"
    if output.exists(): raise RuntimeError("refusing to overwrite intervention cache")
    dev_split, shadow_split = load_search_split(root)
    allowed = set(dev_split["ordered_sample_ids"]); shadow = set(shadow_split["ordered_sample_ids"])
    state_rows = read_jsonl(output_dir / "generator_state_features.jsonl")
    assert_only_search_dev([row["sample_id"] for row in state_rows], allowed,
                           "intervention state input")
    if {row["sample_id"] for row in state_rows} & shadow:
        raise RuntimeError("shadow entered intervention cache")
    provisional = {(row["sample_id"], tuple(row["selected_packet_ids"])):
                   row["provisional_answer"] for row in state_rows}
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    grouped = defaultdict(list)
    for row in labels.rows:
        key = (row["sample_id"], tuple(row["selected_packet_ids"]))
        if key in provisional: grouped[key].append(row)
    parent = ControllerFeatureCache(args.feature_cache)
    cache = SubsetFeatureCache(parent, dev_split["ordered_sample_ids"])
    records = {cache[index]["sample_id"]: cache[index] for index in range(len(cache))}
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(parent, args.static_checkpoint,
                                             args.static_score_cache, device)
    tokenizer, generator, xrag_id, hashes = initialize_generator(args.k2_training_config, device)
    output_dir.mkdir(parents=True, exist_ok=True); written = 0
    sanitized_count = 0
    with output.open("w") as stream:
        for state_index, ((sid, selected), candidates) in enumerate(sorted(grouped.items()), 1):
            scores = static_scores[sid]
            static_top6 = sorted(range(records[sid]["packet_count"]),
                                 key=lambda index: (-float(scores[index]), index))[:6]
            allowed_candidates = {row["candidate_packet_id"]: row for row in candidates
                                  if row["candidate_packet_id"] in static_top6}
            ordered = [packet_id for packet_id in static_top6 if packet_id in allowed_candidates and
                       packet_id not in set(selected)][:4]
            if not ordered: continue
            groups = [list(selected)] + [list(selected) + [packet_id] for packet_id in ordered]
            answer = sanitize_provisional_answer(provisional[(sid, selected)])
            sanitized_count += answer != provisional[(sid, selected)]
            nlls = gold_answer_nll_batch(
                generator, tokenizer, xrag_id, records[sid]["question"],
                answer, records[sid]["packet_embeddings"], groups, device)
            for packet_id, candidate_nll in zip(ordered, nlls[1:]):
                row = {"sample_id": sid, "selected_packet_ids": list(selected),
                       "candidate_packet_id": packet_id,
                       "base_self_nll": nlls[0], "candidate_self_nll": candidate_nll,
                       "self_likelihood_shift": nlls[0] - candidate_nll}
                assert_no_inference_leakage(row, "candidate intervention cache row")
                stream.write(json.dumps(row, ensure_ascii=False) + "\n"); written += 1
            stream.flush()
            if state_index % 160 == 0 or state_index == len(grouped):
                print(json.dumps({"states": state_index, "total": len(grouped),
                                  "candidate_probes": written}), flush=True)
    manifest = {"status": "complete", "format": "packet-xrag-intervention-v1",
                "split": "SEARCH_DEV", "search_dev_hash": dev_split["sha256"],
                "states_considered": len(grouped), "candidate_probe_count": written,
                "sanitized_provisional_answer_states": sanitized_count,
                "initial_candidate_pool": 6, "candidate_probes_per_state": 4,
                "checkpoint_hashes": hashes, "contains_gold_inference_fields": False,
                "search_shadow_accessed": False, "final_100_accessed": False}
    (output_dir / "candidate_intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
