"""Disk-sharded, memory-mapped SFR token-state cache."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


def packet_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_raw(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


class TokenStateShardWriter:
    def __init__(self, output_dir, hidden_size=4096, target_bytes=1024**3):
        self.output_dir = Path(output_dir)
        self.hidden_size = hidden_size
        self.target_bytes = target_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.shards = []
        self._hidden = []
        self._pooled = []
        self._ids = []
        self._masks = []
        self._metadata = []
        self._bytes = 0

    def add(self, key, hidden, pooled, input_ids, mask, metadata):
        if key in self.records:
            raise ValueError(f"duplicate packet key: {key}")
        hidden = hidden.detach().cpu().to(torch.bfloat16).contiguous()
        pooled = pooled.detach().cpu().to(torch.bfloat16).contiguous()
        input_ids = input_ids.detach().cpu().to(torch.int32).contiguous()
        mask = mask.detach().cpu().to(torch.bool).contiguous()
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError("invalid hidden-state shape")
        if len(input_ids) != len(mask) or len(input_ids) != hidden.shape[0]:
            raise ValueError("token fields have inconsistent lengths")
        if pooled.shape != (self.hidden_size,):
            raise ValueError("invalid pooled embedding shape")
        record_bytes = hidden.numel() * 2 + pooled.numel() * 2 + input_ids.numel() * 4 + mask.numel()
        if self._hidden and self._bytes + record_bytes > self.target_bytes:
            self.flush()
        local_index = len(self._metadata)
        self._hidden.append(hidden)
        self._pooled.append(pooled)
        self._ids.append(input_ids)
        self._masks.append(mask)
        self._metadata.append({"key": key, "length": hidden.shape[0], **metadata})
        self._bytes += record_bytes
        self.records[key] = {"shard": len(self.shards), "index": local_index}

    def flush(self):
        if not self._hidden:
            return
        shard_id = len(self.shards)
        prefix = self.output_dir / f"shard-{shard_id:05d}"
        lengths = [item.shape[0] for item in self._hidden]
        token_offsets = []
        cursor = 0
        for length in lengths:
            token_offsets.append(cursor)
            cursor += length
        hidden = torch.cat(self._hidden, dim=0)
        pooled = torch.stack(self._pooled)
        ids = torch.cat(self._ids)
        masks = torch.cat(self._masks)
        _write_raw(prefix.with_suffix(".hidden.bf16"), hidden)
        _write_raw(prefix.with_suffix(".pooled.bf16"), pooled)
        _write_raw(prefix.with_suffix(".ids.i32"), ids)
        _write_raw(prefix.with_suffix(".mask.bool"), masks)
        metadata = []
        for item, offset in zip(self._metadata, token_offsets):
            metadata.append({**item, "token_offset": offset})
        meta_path = prefix.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "hidden_size": self.hidden_size,
            "num_records": len(metadata),
            "num_tokens": cursor,
            "records": metadata,
        }, indent=2) + "\n")
        self.shards.append({
            "id": shard_id,
            "prefix": prefix.name,
            "num_records": len(metadata),
            "num_tokens": cursor,
            "bytes": sum(path.stat().st_size for path in [
                prefix.with_suffix(".hidden.bf16"), prefix.with_suffix(".pooled.bf16"),
                prefix.with_suffix(".ids.i32"), prefix.with_suffix(".mask.bool"), meta_path,
            ]),
        })
        self._hidden.clear(); self._pooled.clear(); self._ids.clear(); self._masks.clear()
        self._metadata.clear(); self._bytes = 0

    def close(self, extra_manifest=None):
        self.flush()
        manifest = {
            "format": "packet-xrag-token-state-mmap-v1",
            "hidden_size": self.hidden_size,
            "records": self.records,
            "shards": self.shards,
            **(extra_manifest or {}),
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest


class TokenStateCache:
    def __init__(self, cache_dir, max_open_shards=2):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.hidden_size = self.manifest["hidden_size"]
        self.max_open_shards = max_open_shards
        self._open = OrderedDict()

    def __contains__(self, key):
        return key in self.manifest["records"]

    def _load_shard(self, shard_id):
        if shard_id in self._open:
            value = self._open.pop(shard_id)
            self._open[shard_id] = value
            return value
        info = self.manifest["shards"][shard_id]
        prefix = self.cache_dir / info["prefix"]
        metadata = json.loads(prefix.with_suffix(".json").read_text())
        num_tokens, num_records = metadata["num_tokens"], metadata["num_records"]
        value = {
            "metadata": metadata["records"],
            "hidden": torch.from_file(str(prefix.with_suffix(".hidden.bf16")), shared=False,
                                      size=num_tokens * self.hidden_size, dtype=torch.bfloat16)
                           .view(num_tokens, self.hidden_size),
            "pooled": torch.from_file(str(prefix.with_suffix(".pooled.bf16")), shared=False,
                                      size=num_records * self.hidden_size, dtype=torch.bfloat16)
                           .view(num_records, self.hidden_size),
            "ids": torch.from_file(str(prefix.with_suffix(".ids.i32")), shared=False,
                                   size=num_tokens, dtype=torch.int32),
            "mask": torch.from_file(str(prefix.with_suffix(".mask.bool")), shared=False,
                                    size=num_tokens, dtype=torch.bool),
        }
        self._open[shard_id] = value
        while len(self._open) > self.max_open_shards:
            self._open.popitem(last=False)
        return value

    def get(self, key):
        location = self.manifest["records"][key]
        shard = self._load_shard(location["shard"])
        metadata = shard["metadata"][location["index"]]
        start, length = metadata["token_offset"], metadata["length"]
        stop = start + length
        return {
            "hidden": shard["hidden"][start:stop],
            "pooled": shard["pooled"][location["index"]],
            "input_ids": shard["ids"][start:stop],
            "mask": shard["mask"][start:stop],
            "metadata": metadata,
        }

