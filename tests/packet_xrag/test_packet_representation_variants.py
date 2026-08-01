from scripts.packet_xrag.run_packet_representation_ablation import build_variant_chunks, exact_minimum_set_cover


def sample():
    return {"context": {"title": ["A", "B"], "sentences": [["a0", "a1", "a2"], ["b0", "b1"]]},
            "supporting_facts": {"title": ["A", "A", "B"], "sent_id": [0, 2, 1]}}


def test_v1_v2_formats_and_gold_coverage():
    for variant, expected in [("V1_TITLE_SENTENCE", "[A] a0"), ("V2_SENTENCE_ONLY", "a0")]:
        chunks, gold = build_variant_chunks(sample(), variant)
        assert chunks[0]["encoder_text"] == expected
        selected = exact_minimum_set_cover(chunks, gold)
        assert len(selected) == 3


def test_window_boundaries_and_document():
    v3, _ = build_variant_chunks(sample(), "V3_LOCAL_WINDOW_3")
    assert v3[0]["encoder_text"] == "[A]\na0\na1"
    assert v3[0]["covered_sentence_ids"] == [0, 1]
    v4, _ = build_variant_chunks(sample(), "V4_FORWARD_WINDOW_2")
    assert v4[-1]["encoder_text"] == "[B]\nb1"
    v5, _ = build_variant_chunks(sample(), "V5_SUPPORT_DOCUMENT")
    assert [c["encoder_text"] for c in v5] == ["[A]\na0 a1 a2", "[B]\nb0 b1"]


def test_minimum_set_cover_and_deterministic_tie_break():
    chunks, gold = build_variant_chunks(sample(), "V3_LOCAL_WINDOW_3")
    first = exact_minimum_set_cover(chunks, gold)
    assert first == exact_minimum_set_cover(chunks, gold)
    coverage = set().union(*(set(chunks[i]["covered_gold_fact_ids"]) for i in first))
    assert coverage == gold
    assert len(first) == 2
