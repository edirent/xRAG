from scripts.packet_xrag.run_k2_selector_benchmark import (
    make_candidate_packets,
    oracle_selection,
)


def sample():
    return {
        "context": {
            "title": ["A", "B"],
            "sentences": [["zero", "one"], ["two", "three"]],
        },
        "supporting_facts": {"title": ["B", "A", "B"], "sent_id": [1, 0, 0]},
    }


def test_gold_mapping_and_original_support_order():
    packets, gold = make_candidate_packets(sample())
    assert [packets[index]["encoder_text"] for index in gold] == ["[B] three", "[A] zero", "[B] two"]
    assert len({packet["packet_id"] for packet in packets}) == len(packets)
    assert all(packets[index]["is_supporting"] for index in gold)


def test_oracle_budgets_and_full_coverage():
    gold = [3, 0, 2]
    assert oracle_selection(gold) == gold
    assert oracle_selection(gold, 1) == [3]
    assert oracle_selection(gold, 2) == [3, 0]


def test_missing_support_fails():
    broken = sample(); broken["supporting_facts"] = {"title": ["missing"], "sent_id": [0]}
    try:
        make_candidate_packets(broken)
        assert False
    except ValueError as error:
        assert "gold mapping failed" in str(error)

