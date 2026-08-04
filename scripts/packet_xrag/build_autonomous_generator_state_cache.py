#!/usr/bin/env python
"""Build one sealed, deployable generator-state cache for SEARCH_DEV states."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from src.packet_xrag.controller.autonomous_features import repetition_fraction
from src.packet_xrag.controller.autonomous_search import (
    SubsetFeatureCache, assert_no_inference_leakage, assert_only_search_dev,
    load_search_split,
)
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.controller.utility_label_dataset import ShardedUtilityLabelDataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/controller/autonomous_search")
    parser.add_argument("--feature-cache", default="cache/controller/features/internal_dev_features")
    parser.add_argument("--labels-root", default="cache/controller/utility_predictor/labels")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args(argv)


@torch.inference_mode()
def generate_features(tokenizer, generator, xrag_id, states, device, max_new_tokens):
    prompts = [v1.build_prompt(item["question"], len(item["selected"]) * 2) for item in states]
    tokenized = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                          padding=True).to(device)
    groups = [item["embeddings"][item["selected"]] for item in states if item["selected"]]
    retrieval = torch.cat(groups).to(device) if groups else None
    expected = sum(len(item["selected"]) * 2 for item in states)
    if int((tokenized.input_ids == xrag_id).sum()) != expected:
        raise RuntimeError("generator state XRAG count mismatch")
    output = generator.generate(
        input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask,
        retrieval_embeds=retrieval, do_sample=False, max_new_tokens=max_new_tokens,
        use_cache=True, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=True,
    )
    sequences = output.sequences
    if sequences.shape[1] > len(output.scores):
        generated = sequences[:, -len(output.scores):]
    else:
        generated = sequences
    rows = []
    for index, item in enumerate(states):
        eos_positions = (generated[index] == tokenizer.eos_token_id).nonzero(as_tuple=False)
        length = int(eos_positions[0]) + 1 if len(eos_positions) else generated.shape[1]
        chosen_logprobs, entropies, margins, eos_probs = [], [], [], []
        for step in range(length):
            logits = output.scores[step][index].float()
            log_probs = F.log_softmax(logits, dim=-1)
            token_id = int(generated[index, step])
            chosen_logprobs.append(float(log_probs[token_id]))
            probabilities = log_probs.exp()
            entropies.append(float(-(probabilities * log_probs).sum()))
            top2 = torch.topk(log_probs, 2).values
            margins.append(float(top2[0] - top2[1]))
            eos_probs.append(float(probabilities[tokenizer.eos_token_id]))
        token_ids = [int(value) for value in generated[index, :length]]
        raw = tokenizer.decode(token_ids, skip_special_tokens=False)
        short = selector.extract_short_answer(raw) or "[EMPTY]"
        row = {"sample_id": item["sample_id"],
               "selected_packet_ids": list(item["selected"]),
               "provisional_answer": short,
               "prompt_tokens": int(tokenized.attention_mask[index].sum()),
               "mean_token_logprob": sum(chosen_logprobs) / len(chosen_logprobs),
               "minimum_token_logprob": min(chosen_logprobs),
               "mean_token_entropy": sum(entropies) / len(entropies),
               "mean_top1_top2_margin": sum(margins) / len(margins),
               "mean_eos_probability": sum(eos_probs) / len(eos_probs),
               "generated_length": length,
               "empty_indicator": int(short == "[EMPTY]"),
               "repetition_fraction": repetition_fraction(token_ids)}
        assert_no_inference_leakage(row, "generator state cache row")
        rows.append(row)
    return rows


def main(argv=None):
    args = parse_args(argv); root = Path(args.root); output_dir = root / "stage1"
    output = output_dir / "generator_state_features.jsonl"
    if output.exists(): raise RuntimeError("refusing to overwrite generator state cache")
    dev_split, shadow_split = load_search_split(root)
    labels = ShardedUtilityLabelDataset(args.labels_root, "internal_dev")
    allowed = set(dev_split["ordered_sample_ids"]); shadow = set(shadow_split["ordered_sample_ids"])
    grouped = defaultdict(list)
    for row in labels.rows:
        if row["sample_id"] in allowed:
            grouped[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(row)
    assert_only_search_dev([key[0] for key in grouped], allowed, "generator state labels")
    if {key[0] for key in grouped} & shadow: raise RuntimeError("shadow entered state cache")
    parent = ControllerFeatureCache(args.feature_cache)
    cache = SubsetFeatureCache(parent, dev_split["ordered_sample_ids"])
    records = {cache[index]["sample_id"]: cache[index] for index in range(len(cache))}
    states = [{"sample_id": sid, "selected": list(selected),
               "question": records[sid]["question"],
               "embeddings": records[sid]["packet_embeddings"]}
              for sid, selected in sorted(grouped)]
    device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer, generator, xrag_id, hashes = initialize_generator(args.k2_training_config, device)
    output_dir.mkdir(parents=True, exist_ok=True); written = 0
    with output.open("w") as stream:
        for start in range(0, len(states), args.batch_size):
            rows = generate_features(tokenizer, generator, xrag_id,
                                     states[start:start + args.batch_size], device,
                                     args.max_new_tokens)
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush(); written += len(rows)
            if written % 160 == 0 or written == len(states):
                print(json.dumps({"generator_states": written, "total": len(states)}), flush=True)
    manifest = {"status": "complete", "format": "packet-xrag-generator-state-v1",
                "split": "SEARCH_DEV", "search_dev_hash": dev_split["sha256"],
                "state_count": len(states), "checkpoint_hashes": hashes,
                "max_new_tokens": args.max_new_tokens, "contains_gold_inference_fields": False,
                "search_shadow_accessed": False, "final_100_accessed": False}
    (output_dir / "generator_state_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
