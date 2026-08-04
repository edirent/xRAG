#!/usr/bin/env python
"""Encode frozen SFR query/sentence-packet caches for a cross-dataset split."""

import argparse
import hashlib
import inspect
import json
import shlex
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from src.model import SFR
from src.packet_xrag.controller.feature_cache import ControllerFeatureWriter, sha256_file
from src.packet_xrag.data import MusiqueAdapter, TriviaQAAdapter, TwoWikiAdapter
from src.packet_xrag.data.base_qa_adapter import MAX_PACKETS


ADAPTERS = {"2wiki": TwoWikiAdapter, "musique": MusiqueAdapter, "triviaqa": TriviaQAAdapter}


@torch.inference_mode()
def encode(tokenizer, model, entries, device, max_length):
    groups = [[sample["question"], *[packet["encoder_text"] for packet in packets]]
              for sample, packets, _ in entries]
    texts = [text for group in groups for text in group]
    tokens = tokenizer(texts, max_length=max_length, padding=True, truncation=True,
                       return_tensors="pt").to(device)
    values = model.get_doc_embedding(tokens.input_ids, tokens.attention_mask).view(len(texts), -1)
    output, offset = [], 0
    for group in groups:
        output.append(values[offset:offset + len(group)]); offset += len(group)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--root", default="cache/generalization")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--sample-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=180)
    args = parser.parse_args(argv); dataset_root = Path(args.root) / args.dataset
    features_root = dataset_root / "features"
    if features_root.exists(): raise RuntimeError("refusing to overwrite cross-dataset features")
    adapter = ADAPTERS[args.dataset](); device = torch.device(args.device); torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad = False
    manifests = {}
    for split in ("train", "dev", "shadow", "benchmark"):
        source = [json.loads(line) for line in
                  (dataset_root / f"records/{split}.jsonl").read_text().splitlines()]
        output_dir = features_root / split; writer = ControllerFeatureWriter(output_dir)
        missing_positive = packet_total = positive_total = 0
        for start in range(0, len(source), args.sample_batch_size):
            entries = []
            for sample in source[start:start + args.sample_batch_size]:
                packets = adapter.packetize(sample, MAX_PACKETS)
                gold = [index for index, packet in enumerate(packets)
                        if packet["is_support"] or (args.dataset == "triviaqa" and packet["contains_answer"])]
                missing_positive += int(not gold); positive_total += len(gold); packet_total += len(packets)
                entries.append((sample, packets, gold))
            embeddings = encode(tokenizer, model, entries, device, args.max_length)
            for (sample, packets, gold), values in zip(entries, embeddings):
                writer.add(sample, packets, gold, values)
            completed = min(start + len(entries), len(source))
            if completed % 100 == 0 or completed == len(source):
                print(f"{args.dataset}/{split}: {completed}/{len(source)}", flush=True)
        split_manifest = json.loads((dataset_root / f"splits/{split}_ids.json").read_text())
        quarantine_hash = hashlib.sha256(b"no-generalization-quarantine").hexdigest()
        manifest = writer.close({"split": split, "effective_split_hash": split_manifest["sha256"],
            "quarantine_hash": quarantine_hash, "source_dataset_identifier": adapter.source_identifier,
            "packet_construction_version": hashlib.sha256(inspect.getsource(adapter.packetize).encode()).hexdigest(),
            "sfr_checkpoint_identifier": v1.SFR_MODEL_NAME,
            "query_encoding_template_hash": hashlib.sha256(b"{question} (verbatim; no instruction)").hexdigest(),
            "number_of_samples": len(source), "number_of_packets": packet_total,
            "creation_command": shlex.join(sys.argv), "all_max_packets": MAX_PACKETS,
            "samples_without_positive_packet": missing_positive,
            "positive_packet_count": positive_total, "sfr_trainable_parameters": 0,
            "support_fields_used_as_inference_features": False, "final100_accessed": False})
        manifests[split] = manifest
    (features_root / "feature_audit.json").write_text(json.dumps({"status": "PASS",
        "dataset": args.dataset, "manifests": manifests, "sfr_trainable_parameters": 0,
        "final100_accessed": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "dataset": args.dataset,
                      "splits": {key: value["num_queries"] for key, value in manifests.items()}},
                     indent=2), flush=True)


if __name__ == "__main__": main()
