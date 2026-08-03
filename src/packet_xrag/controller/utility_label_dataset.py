"""Sharded generator-utility labels and deterministic full-cache states."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


LABEL_FORMAT = "packet-xrag-generator-utility-labels-v1"
INDEX_FORMAT = "packet-xrag-generator-utility-index-v1"
MAX_STATES = 5


def deduplicate_states(raw_states: Sequence[tuple[str, Sequence[int]]], gold_ids: Sequence[int]):
    """Deduplicate by ordered selected tuple and merge source tags."""
    gold = set(gold_ids)
    merged = {}
    for source_tag, selected_values in raw_states:
        selected = list(selected_values)
        if len(selected) != len(set(selected)):
            raise ValueError("state contains duplicate packet IDs")
        key = tuple(selected)
        if key not in merged:
            merged[key] = {
                "state_id": source_tag,
                "state_source_tags": [],
                "selected_packet_ids": selected,
            }
        merged[key]["state_source_tags"].append(source_tag)
    states = []
    for state in merged.values():
        selected = set(state["selected_packet_ids"])
        state["state_source_tags"] = sorted(state["state_source_tags"])
        state["state_gold_recall"] = len(selected & gold) / len(gold)
        state["state_full_gold_support"] = gold.issubset(selected)
        state["num_selected"] = len(selected)
        states.append(state)
    return states


def build_full_state_pool(record: dict, static_ranking: Sequence[int], oracle_packet_id: int | None):
    """Build the locked states, resolving the six-source/five-state conflict.

    S0, S_WRONG1, S_SUFFICIENT, S_GOLD1, and positive-utility S_ORACLE1
    have priority.  A distinct S_STATIC1 is omitted only when retaining it would
    exceed the explicit five-unique-state cap.
    """
    gold = list(dict.fromkeys(record["gold_packet_ids"]))
    gold_set = set(gold)
    wrong = next((packet_id for packet_id in static_ranking if packet_id not in gold_set), None)
    if wrong is None:
        wrong = next((packet_id for packet_id in record["topk_ranking"] if packet_id not in gold_set), None)
    raw = [("S0_EMPTY", []), ("S_GOLD1", [gold[0]]),
           ("S_STATIC1", [static_ranking[0]])]
    if wrong is not None:
        raw.append(("S_WRONG1", [wrong]))
    raw.append(("S_SUFFICIENT", gold))
    if oracle_packet_id is not None:
        raw.append(("S_ORACLE1", [oracle_packet_id]))
    states = deduplicate_states(raw, gold)
    omitted = []
    if len(states) > MAX_STATES:
        removable = next((state for state in states
                          if state["state_source_tags"] == ["S_STATIC1"]), None)
        if removable is None:
            raise RuntimeError("six unique states but no distinct S_STATIC1 to omit")
        states.remove(removable)
        omitted.append("S_STATIC1_five_state_cap")
    if len(states) > MAX_STATES:
        raise RuntimeError("full utility state pool exceeds five unique states")
    required = {"S0_EMPTY", "S_WRONG1", "S_SUFFICIENT"}
    present = {tag for state in states for tag in state["state_source_tags"]}
    if not required.issubset(present):
        raise RuntimeError(f"mandatory state missing: {sorted(required - present)}")
    if oracle_packet_id is not None and "S_ORACLE1" not in present:
        raise RuntimeError("positive S0 oracle state was not retained")
    return states, omitted


def validate_manifest(manifest: dict):
    required = {
        "format", "completion_status", "split", "split_hash", "quarantine_hash",
        "checkpoint_hashes", "prompt_hash", "answer_mask_hash", "candidate_pool_config",
        "state_pool_config", "sample_count", "state_count", "label_count", "shards",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"utility label manifest missing fields: {sorted(missing)}")
    if manifest["format"] != LABEL_FORMAT or manifest["completion_status"] != "complete":
        raise ValueError("utility label cache is incomplete or unsupported")
    return True


class ShardedUtilityLabelDataset(Dataset):
    """Strict reader that refuses incomplete label caches."""

    def __init__(self, labels_root, split, load_rows=True):
        self.labels_root = Path(labels_root)
        self.split = split
        self.manifest = json.loads(
            (self.labels_root / f"{split}_manifest.json").read_text()
        )
        validate_manifest(self.manifest)
        self.index = json.loads((self.labels_root / split / "index.json").read_text())
        if self.index.get("format") != INDEX_FORMAT:
            raise ValueError("unsupported utility label index")
        if self.index.get("sample_count") != self.manifest["sample_count"]:
            raise ValueError("utility label index sample count mismatch")
        self.samples = self.index["samples"]
        self.sample_by_id = {item["sample_id"]: item for item in self.samples}
        if len(self.sample_by_id) != len(self.samples):
            raise ValueError("duplicate sample IDs in utility label index")
        self.rows = []
        if load_rows:
            for shard in self.manifest["shards"]:
                path = self.labels_root / split / shard["filename"]
                shard_rows = [json.loads(line) for line in path.read_text().splitlines()
                              if line.strip()]
                if len(shard_rows) != shard["label_count"]:
                    raise ValueError(f"utility label shard count mismatch: {path}")
                self.rows.extend(shard_rows)
            if len(self.rows) != self.manifest["label_count"]:
                raise ValueError("utility label total count mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def state_groups(self):
        groups = defaultdict(list)
        for index, row in enumerate(self.rows):
            groups[(row["sample_id"], tuple(row["selected_packet_ids"]))].append(index)
        return list(groups.values())


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def utility_target_statistics(train_deltas: Sequence[float]):
    absolute = [abs(float(value)) for value in train_deltas]
    clip = max(0.1, percentile(absolute, .99))
    quantiles = {f"p{int(q * 100):02d}": percentile(train_deltas, q)
                 for q in (.01, .05, .25, .50, .75, .95, .99)}
    return {
        "utility_clip_value": clip,
        "clip_definition": "max(0.1, train P99(abs(delta_utility)))",
        "training_label_count": len(train_deltas),
        "training_utility_quantiles": quantiles,
        "internal_dev_used": False, "benchmark_used": False,
    }


def normalize_delta(delta: torch.Tensor, clip_value: float):
    return delta.clamp(-clip_value, clip_value) / clip_value
