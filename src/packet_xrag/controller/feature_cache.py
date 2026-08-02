"""Deterministic controller splits and memory-mapped frozen SFR features."""

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


FORMAT = "packet-xrag-controller-features-v1"
SPLIT_SEED = 20260803
HIDDEN_SIZE = 4096


def sample_id(sample):
    return str(sample.get("id", sample.get("_id")))


def ordered_ids_sha256(ids):
    return hashlib.sha256("".join(ids).encode("utf-8")).hexdigest()


def normalized_question(question):
    text = unicodedata.normalize("NFKC", question).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def question_sha256(question, normalized=False):
    text = normalized_question(question) if normalized else question
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_records(records, seed=SPLIT_SEED, train_count=4500):
    """Shuffle the locked source IDs once, retaining that order in both splits."""
    records_by_id = {sample_id(item[0]): item for item in records}
    if len(records_by_id) != len(records):
        raise ValueError("duplicate sample IDs in controller source pool")
    ids = list(records_by_id)
    random.Random(seed).shuffle(ids)
    if len(ids) != 5000 or train_count != 4500:
        raise ValueError("locked controller split requires exactly 5,000 -> 4,500/500")
    train_ids, dev_ids = ids[:train_count], ids[train_count:]
    return ([records_by_id[sid] for sid in train_ids],
            [records_by_id[sid] for sid in dev_ids])


def overlap_audit(named_records):
    """Return all pairwise ID, exact-question, and normalized-question overlaps."""
    views = {}
    for name, records in named_records.items():
        ids = [sample_id(item[0]) for item in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample IDs within {name}")
        views[name] = {
            "ids": set(ids),
            "exact": {question_sha256(item[0]["question"]) for item in records},
            "normalized": {question_sha256(item[0]["question"], True) for item in records},
        }
    overlaps = {}
    names = list(named_records)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            key = f"{left}__{right}"
            overlaps[key] = {
                "sample_id_overlap": sorted(views[left]["ids"] & views[right]["ids"]),
                "exact_question_overlap": sorted(views[left]["exact"] & views[right]["exact"]),
                "normalized_question_overlap": sorted(
                    views[left]["normalized"] & views[right]["normalized"]
                ),
            }
    return overlaps


def assert_no_overlap(overlaps):
    failures = {pair: fields for pair, fields in overlaps.items()
                if any(fields.values())}
    if failures:
        raise RuntimeError(f"controller data isolation failed: {failures}")


def make_candidate_packets(sample):
    supporting_pairs = list(zip(
        sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"]
    ))
    supporting_set = set(supporting_pairs)
    packets = []
    for doc_id, (title, sentences) in enumerate(zip(
            sample["context"]["title"], sample["context"]["sentences"])):
        for sentence_id, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            packets.append({
                "packet_id": len(packets),
                "doc_id": doc_id,
                "title": title,
                "sentence_id": sentence_id,
                "text": sentence,
                "encoder_text": f"[{title}] {sentence}",
                "is_supporting": (title, sentence_id) in supporting_set,
            })
    pair_to_ids = defaultdict(list)
    for packet in packets:
        pair_to_ids[(packet["title"], packet["sentence_id"])].append(packet["packet_id"])
    missing, ambiguous, gold_ids = [], [], []
    for pair in supporting_pairs:
        matches = pair_to_ids.get(pair, [])
        if not matches:
            missing.append(pair)
        elif len(matches) != 1:
            ambiguous.append((pair, matches))
        elif matches[0] not in gold_ids:
            gold_ids.append(matches[0])
    if missing or ambiguous:
        raise ValueError(f"gold mapping failed; missing={missing}, ambiguous={ambiguous}")
    if not packets or not gold_ids:
        raise ValueError("empty candidate or gold packet set")
    if set(gold_ids) != {packet["packet_id"] for packet in packets if packet["is_supporting"]}:
        raise ValueError("gold support mapping is incomplete")
    return packets, gold_ids


def ranking_features(embeddings, mmr_lambda=0.5):
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("expected one query and at least one packet embedding")
    normalized = F.normalize(embeddings.float(), dim=-1)
    relevance = (normalized[1:] @ normalized[0]).cpu()
    similarities = (normalized[1:] @ normalized[1:].T).cpu()
    topk = sorted(range(len(relevance)), key=lambda i: (-float(relevance[i]), i))
    selected, remaining = [], set(topk)
    maximum_similarity = None
    while remaining:
        choices = []
        for index in remaining:
            redundancy = float(maximum_similarity[index]) if selected else 0.0
            score = (float(relevance[index]) if not selected else
                     mmr_lambda * float(relevance[index]) -
                     (1.0 - mmr_lambda) * redundancy)
            choices.append((-score, -float(relevance[index]), index))
        chosen = min(choices)[2]
        selected.append(chosen)
        remaining.remove(chosen)
        maximum_similarity = (similarities[:, chosen].clone() if maximum_similarity is None
                              else torch.maximum(maximum_similarity, similarities[:, chosen]))
    return relevance, topk, selected


def _write_tensor(stream, tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    stream.write(raw)


class ControllerFeatureWriter:
    """Append-only writer for query and packet BF16 embedding mmaps."""

    def __init__(self, output_dir, hidden_size=HIDDEN_SIZE):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size
        self.query_stream = (self.output_dir / "query_embeddings.bf16").open("wb")
        self.packet_stream = (self.output_dir / "packet_embeddings.bf16").open("wb")
        self.record_stream = (self.output_dir / "records.jsonl").open("w")
        self.num_queries = self.num_packets = 0

    def add(self, sample, packets, gold_ids, embeddings):
        if embeddings.shape != (len(packets) + 1, self.hidden_size):
            raise ValueError(f"unexpected SFR embedding shape: {tuple(embeddings.shape)}")
        relevance, topk, mmr = ranking_features(embeddings)
        embeddings = embeddings.detach().cpu().to(torch.bfloat16).contiguous()
        packet_offset = self.num_packets
        _write_tensor(self.query_stream, embeddings[0])
        _write_tensor(self.packet_stream, embeddings[1:])
        record = {
            "sample_id": sample_id(sample),
            "question": sample["question"],
            "answer": sample["answer"],
            "query_index": self.num_queries,
            "packet_offset": packet_offset,
            "packet_count": len(packets),
            "gold_packet_ids": gold_ids,
            "topk_scores": [float(value) for value in relevance],
            "topk_ranking": topk,
            "mmr_ranking": mmr,
            "packets": packets,
        }
        self.record_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.num_queries += 1
        self.num_packets += len(packets)
        return record

    def close(self, extra_manifest=None):
        self.query_stream.close()
        self.packet_stream.close()
        self.record_stream.close()
        manifest = {
            "format": FORMAT,
            "hidden_size": self.hidden_size,
            "dtype": "bfloat16",
            "num_queries": self.num_queries,
            "num_packets": self.num_packets,
            "query_protocol": "{question} (verbatim; no instruction)",
            "mmr_lambda": 0.5,
            **(extra_manifest or {}),
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest


class ControllerFeatureCache:
    """Read-only views over a complete controller feature cache."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if self.manifest["format"] != FORMAT:
            raise ValueError("unsupported controller feature cache format")
        self.hidden_size = self.manifest["hidden_size"]
        self.records = [json.loads(line) for line in
                        (self.cache_dir / "records.jsonl").read_text().splitlines()]
        if len(self.records) != self.manifest["num_queries"]:
            raise ValueError("feature-cache record count mismatch")
        self.queries = torch.from_file(
            str(self.cache_dir / "query_embeddings.bf16"), shared=False,
            size=len(self.records) * self.hidden_size, dtype=torch.bfloat16,
        ).view(len(self.records), self.hidden_size)
        self.packets = torch.from_file(
            str(self.cache_dir / "packet_embeddings.bf16"), shared=False,
            size=self.manifest["num_packets"] * self.hidden_size, dtype=torch.bfloat16,
        ).view(self.manifest["num_packets"], self.hidden_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        start = record["packet_offset"]
        stop = start + record["packet_count"]
        return {**record, "query_embedding": self.queries[record["query_index"]],
                "packet_embeddings": self.packets[start:stop]}
