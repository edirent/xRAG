import json

import pytest

from scripts.packet_xrag.bootstrap_packet_representation_ablation import load_paired, paired_bootstrap
from scripts.packet_xrag.run_packet_representation_ablation import HISTORICAL_FIRST_100_F1, OFFICIAL_FULL_500_F1, validate_validation_id_record


def paired_rows(path, missing=False):
    variants = ["V1_TITLE_SENTENCE", "V2_SENTENCE_ONLY", "V3_LOCAL_WINDOW_3", "V4_FORWARD_WINDOW_2", "V5_SUPPORT_DOCUMENT"]
    with path.open("w") as stream:
        for variant in variants:
            for index in range(500 - (1 if missing and variant == variants[-1] else 0)):
                stream.write(json.dumps({"variant": variant, "sample_id": str(index), "short_f1": float(index % 2)}) + "\n")


def test_baseline_scope_labels_are_distinct():
    assert HISTORICAL_FIRST_100_F1 == pytest.approx(63.019047619047605)
    assert OFFICIAL_FULL_500_F1 == pytest.approx(58.589785360838036)
    full_ids = [str(index) for index in range(500)]
    assert full_ids[:100] == [str(index) for index in range(100)]


def test_full_500_id_record_requires_exact_order_and_hash():
    import hashlib
    full_ids = [str(index) for index in range(500)]
    record = {"num_validation_samples": 500, "sample_ids": full_ids,
              "split_hash": hashlib.sha256("".join(full_ids).encode()).hexdigest()}
    assert validate_validation_id_record(record, full_ids) == record["split_hash"]
    changed = dict(record, sample_ids=list(reversed(full_ids)))
    with pytest.raises(AssertionError):
        validate_validation_id_record(changed, full_ids)


def test_variant_sample_sets_must_match(tmp_path):
    path = tmp_path / "missing.jsonl"
    paired_rows(path, missing=True)
    with pytest.raises(ValueError, match="sample_id set differs"):
        load_paired(path)


def test_paired_bootstrap_is_seed_reproducible(tmp_path):
    path = tmp_path / "paired.jsonl"
    paired_rows(path)
    records, _ = load_paired(path)
    first = paired_bootstrap(records, num_bootstrap=50, seed=42)
    second = paired_bootstrap(records, num_bootstrap=50, seed=42)
    assert first == second
    assert all(comparison["delta_vs_v1"] == 0 for comparison in first["comparisons"].values())
