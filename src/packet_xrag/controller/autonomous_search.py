"""Isolation, bookkeeping, and leakage guards for autonomous controller search."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


SEARCH_SEED = 20260804
SEARCH_DEV_COUNT = 350
SEARCH_SHADOW_COUNT = 150
PROBE_SUBSET_COUNT = 150
EXPECTED_INTERNAL_DEV_HASH = "df6ef10179d693759191e2bff3ca2056a1438b4c08d10cef3e8b48357c38f63d"
EXPECTED_BENCHMARK_HASH = "8f925ff8ababf1efc6bb8a913e6d5431437610b0bb30fa8357a57dfbb5f24052"
EXPECTED_TRAIN_HASH = "91a47f422aeebe7f212058330e4c1357651b8c729b62bdec66379df413104840"
QUARANTINED_ID = "5a7b23ca554299042af8f703"

FORBIDDEN_INFERENCE_KEYS = {
    "gold_answer", "gold_answer_tokens", "gold_packet_ids", "gold_supporting_label",
    "true_delta_utility", "delta_utility", "base_answer_nll", "candidate_answer_nll",
    "candidate_is_gold", "is_gold", "full_gold_support", "state_full_gold_support",
}


def ordered_ids_sha256(ids):
    return hashlib.sha256("".join(str(value) for value in ids).encode()).hexdigest()


def split_search_ids(ordered_ids, seed=SEARCH_SEED):
    ids = list(ordered_ids)
    if len(ids) != SEARCH_DEV_COUNT + SEARCH_SHADOW_COUNT or len(ids) != len(set(ids)):
        raise ValueError("search split requires exactly 500 unique internal-dev IDs")
    random.Random(seed).shuffle(ids)
    search_dev = ids[:SEARCH_DEV_COUNT]
    search_shadow = ids[SEARCH_DEV_COUNT:]
    if set(search_dev) & set(search_shadow):
        raise AssertionError("search-dev and search-shadow overlap")
    return search_dev, search_shadow


def fixed_probe_subset(search_dev_ids, seed=SEARCH_SEED):
    ids = list(search_dev_ids)
    if len(ids) != SEARCH_DEV_COUNT or len(ids) != len(set(ids)):
        raise ValueError("probe subset requires the locked 350 search-dev IDs")
    random.Random(seed).shuffle(ids)
    return ids[:PROBE_SUBSET_COUNT]


def assert_only_search_dev(sample_ids, search_dev_ids, context="artifact"):
    actual, allowed = set(sample_ids), set(search_dev_ids)
    leaked = actual - allowed
    if leaked:
        raise RuntimeError(f"{context} contains non-SEARCH_DEV IDs: {sorted(leaked)[:5]}")
    return True


def find_forbidden_inference_keys(payload, prefix=""):
    failures = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in FORBIDDEN_INFERENCE_KEYS:
                failures.append(path)
            failures.extend(find_forbidden_inference_keys(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            failures.extend(find_forbidden_inference_keys(value, f"{prefix}[{index}]"))
    return failures


def assert_no_inference_leakage(payload, context="inference artifact"):
    failures = find_forbidden_inference_keys(payload)
    if failures:
        raise RuntimeError(f"gold-derived inference leakage in {context}: {failures[:10]}")
    return True


class SubsetFeatureCache:
    """Read-only ordered subset view without copying the frozen embeddings."""

    def __init__(self, parent, ordered_ids):
        by_id = {record["sample_id"]: index for index, record in enumerate(parent.records)}
        missing = set(ordered_ids) - set(by_id)
        if missing:
            raise ValueError(f"subset IDs absent from feature cache: {sorted(missing)[:5]}")
        self.parent = parent
        self.indices = [by_id[sid] for sid in ordered_ids]
        self.records = [parent.records[index] for index in self.indices]
        self.hidden_size = parent.hidden_size
        self.manifest = {**parent.manifest, "subset_count": len(self.indices),
                         "subset_hash": ordered_ids_sha256(ordered_ids)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.parent[self.indices[index]]


def load_search_split(root="cache/controller/autonomous_search"):
    root = Path(root) / "splits"
    dev = json.loads((root / "search_dev_ids.json").read_text())
    shadow = json.loads((root / "search_shadow_ids.json").read_text())
    if dev["sample_count"] != SEARCH_DEV_COUNT or shadow["sample_count"] != SEARCH_SHADOW_COUNT:
        raise RuntimeError("autonomous search split cardinality mismatch")
    if ordered_ids_sha256(dev["ordered_sample_ids"]) != dev["sha256"]:
        raise RuntimeError("SEARCH_DEV hash mismatch")
    if ordered_ids_sha256(shadow["ordered_sample_ids"]) != shadow["sha256"]:
        raise RuntimeError("SEARCH_SHADOW hash mismatch")
    if set(dev["ordered_sample_ids"]) & set(shadow["ordered_sample_ids"]):
        raise RuntimeError("autonomous search split overlap")
    return dev, shadow


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

