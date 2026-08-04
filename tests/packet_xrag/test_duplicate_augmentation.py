from src.packet_xrag.composition.diagnostics import duplicate_stress_sets


def test_duplicate_augmentation_keeps_base_and_appends_exact_counts():
    record = {"sample_id": "s", "packet_count": 6, "gold_packet_ids": [0],
              "topk_ranking": [1, 0, 2, 3, 4, 5]}
    base = [1, 0, 2]
    groups = duplicate_stress_sets(record, base)
    for kind in ("GOLD", "NONGOLD", "RANDOM"):
        for count in (1, 2, 4):
            assert groups[f"DUP_{kind}_X{count}"][:3] == base
            assert len(groups[f"DUP_{kind}_X{count}"]) == 3 + count
