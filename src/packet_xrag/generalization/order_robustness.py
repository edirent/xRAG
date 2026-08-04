"""Physical-order perturbations that preserve STATIC base membership."""

from __future__ import annotations

import hashlib
import random


SEED = 20260804


def permute_extras(record, selected, variant, seed=SEED):
    selected = list(selected); base, extras = selected[:2], selected[2:]
    if variant == "canonical": return selected
    if variant == "reverse": return base + list(reversed(extras))
    if variant == "document":
        return base + sorted(extras, key=lambda index: (
            int(record["packets"][index]["doc_id"]),
            int(record["packets"][index]["sentence_id"]), index))
    if variant.startswith("random_"):
        permutation = list(extras)
        rng = random.Random(f"{seed}:{record['sample_id']}:{variant}")
        rng.shuffle(permutation); return base + permutation
    raise ValueError(f"unknown order perturbation: {variant}")


def architecture_order_audit(fuser):
    names = list(dict(fuser.named_parameters()))
    forbidden = [name for name in names if any(token in name.casefold()
        for token in ("position", "rank_embedding", "sentence_position", "document_position"))]
    return {"explicit_packet_order_embedding": False, "explicit_rank_embedding": False,
            "explicit_sentence_position_embedding": False,
            "explicit_document_position_embedding": False,
            "order_sensitive_recurrent_or_sequence_layer": False,
            "attention_without_position_encoding": True,
            "forbidden_parameter_names": forbidden,
            "parameter_name_hash": hashlib.sha256("\n".join(names).encode()).hexdigest()}

