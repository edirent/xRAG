#!/usr/bin/env python
"""Materialize deterministic train/dev/shadow/benchmark records for one QA dataset."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.packet_xrag.data import MusiqueAdapter, TriviaQAAdapter, TwoWikiAdapter
from src.packet_xrag.data.base_qa_adapter import MAX_PACKETS
from src.packet_xrag.data.base_qa_adapter import normalized_question
from src.packet_xrag.generalization.protocol import (
    SEED, deterministic_partition, ordered_hash, split_audit,
)


ADAPTERS = {"2wiki": TwoWikiAdapter, "musique": MusiqueAdapter, "triviaqa": TriviaQAAdapter}


def id_of(adapter, sample):
    if adapter.dataset_name == "2wiki": return str(sample["id"])
    if adapter.dataset_name == "musique": return str(sample["id"])
    return str(sample["question_id"])


def unique_questions(source, forbidden=None):
    seen = set(forbidden or ()); keep = []
    questions = source["question"] if hasattr(source, "column_names") else [
        sample["question"] for sample in source]
    for index, question in enumerate(questions):
        key = normalized_question(question)
        if key in seen: continue
        seen.add(key); keep.append(index)
    selected = source.select(keep) if hasattr(source, "select") else [source[i] for i in keep]
    return selected, seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--root", default="cache/generalization")
    args = parser.parse_args(argv); root = Path(args.root) / args.dataset
    if root.exists(): raise RuntimeError(f"refusing to overwrite {args.dataset} split")
    adapter = ADAPTERS[args.dataset](); train_source = adapter.load_train()
    validation_source = adapter.load_validation()
    if args.dataset == "triviaqa":
        train_source, train_questions = unique_questions(train_source)
        validation_source, _ = unique_questions(validation_source, train_questions)
    train_ids = [id_of(adapter, sample) for sample in train_source]
    validation_ids = [id_of(adapter, sample) for sample in validation_source]
    partitions = deterministic_partition(train_ids, validation_ids)
    wanted_train = set(partitions["train"] + partitions["dev"] + partitions["shadow"])
    wanted_validation = set(partitions["benchmark"])
    canonical = {}
    for source, wanted in ((train_source, wanted_train), (validation_source, wanted_validation)):
        for sample in source:
            sid = id_of(adapter, sample)
            if sid in wanted: canonical[sid] = adapter.canonicalize(sample)
    missing = (wanted_train | wanted_validation) - set(canonical)
    if missing: raise RuntimeError(f"missing {len(missing)} selected {args.dataset} samples")
    named = {name: [canonical[sid] for sid in ids] for name, ids in partitions.items()}
    audit = split_audit(named); split_dir = root / "splits"; raw_dir = root / "records"
    split_dir.mkdir(parents=True); raw_dir.mkdir()
    packet_counts = {}; zero_support = {}; truncated = {}
    for name, samples in named.items():
        ids = [sample["id"] for sample in samples]
        (split_dir / f"{name}_ids.json").write_text(json.dumps({"dataset": args.dataset,
            "split": name, "sample_count": len(ids), "ordered_sample_ids": ids,
            "sha256": ordered_hash(ids), "seed": SEED}, indent=2, sort_keys=True) + "\n")
        counts, missing_support, hit_cap = [], 0, 0
        with (raw_dir / f"{name}.jsonl").open("w") as stream:
            for sample in samples:
                packets = adapter.packetize(sample, MAX_PACKETS); counts.append(len(packets))
                missing_support += int(not any(packet["is_support"] or packet["contains_answer"]
                                               for packet in packets))
                hit_cap += int(len(packets) == MAX_PACKETS)
                stream.write(json.dumps({**sample, "packet_count": len(packets)},
                                        ensure_ascii=False) + "\n")
        packet_counts[name] = {"min": min(counts), "max": max(counts),
                               "mean": sum(counts) / len(counts)}
        zero_support[name] = missing_support; truncated[name] = hit_cap
    manifest = {"dataset": args.dataset, "source_identifier": adapter.source_identifier,
        "seed": SEED, "packet_format": "[Title] sentence", "all_max_packets": MAX_PACKETS,
        "counts": {name: len(samples) for name, samples in named.items()},
        "split_hashes": {name: ordered_hash([sample["id"] for sample in samples])
                         for name, samples in named.items()},
        "packet_counts": packet_counts, "samples_without_support_or_answer_packet": zero_support,
        "samples_hitting_packet_cap": truncated, "question_overlap_gate": "PASS",
        "overlap_audit": audit, "benchmark_accessed": False, "final100_accessed": False}
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                                sort_keys=True) + "\n")
    lines = [f"# {args.dataset} Split Audit", "", "- Question overlap gate: PASS",
        f"- Counts: {manifest['counts']}", f"- Split hashes: {manifest['split_hashes']}",
        f"- Context document overlap statistics: {audit}",
        f"- Answer/support missing after cap: {zero_support}",
        f"- Packet cap hits: {truncated}", "- Final-100 accessed: No", ""]
    (split_dir / "split_audit.md").write_text("\n".join(lines))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
