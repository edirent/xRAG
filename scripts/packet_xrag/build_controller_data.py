#!/usr/bin/env python
"""Build the locked Stage-0 controller split, packet metadata, and SFR caches."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.token_resampler_common import EXPECTED_SPLIT_HASH, locked_records
from src.model import SFR
from src.packet_xrag.controller.feature_cache import (
    ControllerFeatureWriter,
    SPLIT_SEED,
    assert_no_overlap,
    make_candidate_packets,
    ordered_ids_sha256,
    overlap_audit,
    sample_id,
    split_records,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--output-root", default="cache/controller")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_version():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {
        "git_revision": revision,
        "build_script_sha256": source_hash(__file__),
        "feature_cache_sha256": source_hash(REPO_ROOT / "src/packet_xrag/controller/feature_cache.py"),
    }


def write_split_record(path, name, records, seed, version):
    ids = [sample_id(item[0]) for item in records]
    payload = {
        "split": name,
        "sample_count": len(ids),
        "ordered_sample_ids": ids,
        "sha256": ordered_ids_sha256(ids),
        "source_dataset": "hotpotqa/hotpot_qa:distractor:train; locked V1 source pool",
        "seed": seed,
        "creation_code_version": version,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def prepare_splits(args, source_records, benchmark_records, output_root):
    if args.seed != SPLIT_SEED:
        raise ValueError(f"controller split seed is locked to {SPLIT_SEED}")
    train, dev = split_records(source_records, args.seed)
    overlaps = overlap_audit({"train": train, "internal_dev": dev, "benchmark": benchmark_records})
    assert_no_overlap(overlaps)
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    version = code_version()
    train_payload = write_split_record(
        split_dir / "controller_train_ids.json", "controller_train", train, args.seed, version
    )
    dev_payload = write_split_record(
        split_dir / "controller_internal_dev_ids.json", "controller_internal_dev", dev,
        args.seed, version,
    )
    audit = {
        "status": "PASS",
        "source_sample_count": len(source_records),
        "benchmark_sample_count": len(benchmark_records),
        "benchmark_sha256": EXPECTED_SPLIT_HASH,
        "controller_train": train_payload,
        "controller_internal_dev": dev_payload,
        "overlaps": overlaps,
        "final_100_accessed": False,
        "final_100_note": "Not inspected: final-100 access is forbidden by protocol.",
        "creation_code_version": version,
    }
    (split_dir / "controller_split_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Controller Split Audit", "", "- Status: PASS",
        f"- Seed: {args.seed}", f"- Controller train: {len(train)}",
        f"- Internal dev: {len(dev)}", f"- Frozen benchmark: {len(benchmark_records)}",
        f"- Train ordered-ID SHA256: `{train_payload['sha256']}`",
        f"- Internal-dev ordered-ID SHA256: `{dev_payload['sha256']}`",
        f"- Benchmark ordered-ID SHA256: `{EXPECTED_SPLIT_HASH}`",
        "- All pairwise ID/exact-question/normalized-question overlaps: 0",
        "- Final 100: not accessed (protocol barrier)",
    ]
    (split_dir / "controller_split_audit.md").write_text("\n".join(lines) + "\n")
    return train, dev, audit


def preflight_gold_mappings(named_records):
    """Validate every supporting-fact mapping before loading SFR or writing features."""
    failures = []
    for split_name, records in named_records.items():
        for index, (sample, _, _) in enumerate(records):
            try:
                make_candidate_packets(sample)
            except ValueError as error:
                failures.append({
                    "split": split_name,
                    "index": index,
                    "sample_id": sample_id(sample),
                    "question": sample["question"],
                    "error": str(error),
                })
    if failures:
        raise RuntimeError(
            "MANDATORY STOP: controller gold support mapping failed: " +
            json.dumps(failures, ensure_ascii=False)
        )
    return True


@torch.inference_mode()
def encode_samples(tokenizer, model, entries, device, max_length):
    # Exact historical Frozen-K2 ranking protocol: raw question, no instruction.
    groups = [
        [sample["question"]] + [packet["encoder_text"] for packet in packets]
        for sample, packets, _ in entries
    ]
    texts = [text for group in groups for text in group]
    tokens = tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    embeddings = model.get_doc_embedding(
        tokens.input_ids, tokens.attention_mask
    ).view(len(texts), -1)
    outputs, offset = [], 0
    for group in groups:
        outputs.append(embeddings[offset:offset + len(group)])
        offset += len(group)
    assert offset == len(texts)
    return outputs


def build_split_cache(name, records, tokenizer, model, args, output_root):
    feature_dir = output_root / "features" / f"{name}_features"
    packet_path = output_root / "packets" / f"{name}_packets.jsonl"
    writer = ControllerFeatureWriter(feature_dir)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_count = support_count = 0
    with packet_path.open("w") as packet_stream:
        for start in range(0, len(records), args.sample_batch_size):
            entries = []
            for sample, _, _ in records[start:start + args.sample_batch_size]:
                packets, gold_ids = make_candidate_packets(sample)
                entries.append((sample, packets, gold_ids))
            embedding_groups = encode_samples(
                tokenizer, model, entries, args.device, args.max_length
            )
            for (sample, packets, gold_ids), embeddings in zip(entries, embedding_groups):
                feature_record = writer.add(sample, packets, gold_ids, embeddings)
                sid = sample_id(sample)
                for packet in packets:
                    global_offset = feature_record["packet_offset"] + packet["packet_id"]
                    cached = {
                        "sample_id": sid, **packet,
                        "sfr_embedding": (
                            f"{feature_dir.resolve() / 'packet_embeddings.bf16'}"
                            f"#bf16[{global_offset},4096]"
                        ),
                    }
                    packet_stream.write(json.dumps(cached, ensure_ascii=False) + "\n")
                    packet_count += 1
                    support_count += int(packet["is_supporting"])
            completed = min(start + len(entries), len(records))
            if completed % args.log_every == 0 or completed == len(records):
                print(f"{name}: {completed}/{len(records)} samples, {packet_count} packets", flush=True)
    manifest = writer.close({
        "split": name,
        "ordered_ids_sha256": ordered_ids_sha256([sample_id(item[0]) for item in records]),
        "sfr_model_identifier": v1.SFR_MODEL_NAME,
        "sfr_trainable_parameters": 0,
        "max_length": args.max_length,
        "sample_batch_size": args.sample_batch_size,
        "packet_metadata_path": str(packet_path.resolve()),
        "gold_support_count": support_count,
    })
    return {**manifest, "packet_metadata_bytes": packet_path.stat().st_size}


def main(argv=None):
    args = parse_args(argv)
    output_root = Path(args.output_root)
    print("Recovering locked 5,000 source and benchmark-500 records", flush=True)
    source, benchmark = locked_records(args.split_file, args.v1_training_config)
    train, dev, _ = prepare_splits(args, source, benchmark, output_root)
    preflight_gold_mappings({"train": train, "internal_dev": dev, "benchmark": benchmark})
    print("Stage-0 isolation gate passed; loading frozen SFR", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("SFR must remain frozen")
    manifests = {}
    for name, records in (("train", train), ("internal_dev", dev), ("benchmark", benchmark)):
        manifests[name] = build_split_cache(name, records, tokenizer, model, args, output_root)
    packet_audit = {
        "status": "PASS", "gold_mapping_complete": True,
        "splits": {name: {key: value for key, value in manifest.items()
                           if key in ("num_queries", "num_packets", "gold_support_count",
                                      "ordered_ids_sha256", "packet_metadata_bytes")}
                   for name, manifest in manifests.items()},
    }
    (output_root / "packets" / "packet_cache_audit.json").write_text(
        json.dumps(packet_audit, indent=2, sort_keys=True) + "\n"
    )
    feature_audit = [
        "# Controller Feature Cache Audit", "", "- Status: PASS",
        "- SFR: `Salesforce/SFR-Embedding-Mistral` (frozen; 0 trainable parameters)",
        "- Query protocol: raw question verbatim, no instruction",
        "- Embeddings: BF16, 4096 dimensions, memory-mapped",
        "- TOPK: descending query/packet cosine with packet-ID tie break",
        "- MMR: full ranking, lambda=0.5, frozen benchmark tie breaks",
        "- Gold support mapping: complete for every sample",
        "- Final 100: not accessed",
    ]
    for name, manifest in manifests.items():
        feature_audit.append(
            f"- {name}: {manifest['num_queries']} queries, {manifest['num_packets']} packets, "
            f"{manifest['gold_support_count']} gold packets"
        )
    (output_root / "features" / "feature_cache_audit.md").write_text(
        "\n".join(feature_audit) + "\n"
    )
    decision = [
        "# Stage 0 Decision", "", "- Decision: PASS — proceed to Stage 1",
        "- Split isolation: PASS", "- Candidate gold mapping: PASS",
        "- Frozen feature cache: PASS", "- Final 100 accessed: no",
    ]
    (output_root / "stage0_decision.md").write_text("\n".join(decision) + "\n")
    print(json.dumps({"status": "PASS", "splits": packet_audit["splits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
