import copy

from src.packet_xrag.controller.generator_utility import (
    build_candidate_pool, build_state_pool, candidate_addition_groups,
    choose_feasibility_records,
)


def make_record(sample_id="sample"):
    packets = [
        {"packet_id": index, "doc_id": index // 3, "title": f"D{index // 3}",
         "sentence_id": index % 3, "text": f"sentence {index}",
         "is_supporting": index in {1, 4}}
        for index in range(10)
    ]
    return {
        "sample_id": sample_id, "packet_count": len(packets), "packets": packets,
        "gold_packet_ids": [4, 1], "topk_ranking": list(range(9, -1, -1)),
        "mmr_ranking": [3, 4, 5, 6, 7, 8, 9, 2, 1, 0],
        "topk_scores": [index / 10 for index in range(10)],
    }


def test_candidate_pool_is_deterministic_and_preserves_all_gold():
    record = make_record(); static_ranking = list(range(10)); scores = list(range(10))
    first = build_candidate_pool(record, scores, static_ranking)
    second = build_candidate_pool(copy.deepcopy(record), scores, static_ranking)
    assert first == second
    assert len(first) <= 12
    assert set(record["gold_packet_ids"]).issubset({row["packet_id"] for row in first})


def test_state_construction_deduplicates_and_keeps_original_gold1():
    record = make_record()
    states = build_state_pool(record, [4, 0, 1, 2, 3, 5, 6, 7, 8, 9])
    assert len({tuple(sorted(state["selected_packet_ids"])) for state in states}) == len(states)
    merged = next(state for state in states if state["selected_packet_ids"] == [4])
    assert merged["state_source_tags"] == ["S_GOLD1", "S_STATIC1"]
    sufficient = next(state for state in states if "S_SUFFICIENT" in state["state_source_tags"])
    assert sufficient["selected_packet_ids"] == [1, 4]
    assert sufficient["full_gold_support"] is True


def test_candidate_already_selected_is_excluded_and_append_order_is_stable():
    groups = candidate_addition_groups([7, 2], [2, 3, 7, 5])
    assert groups == [[7, 2], [7, 2, 3], [7, 2, 5]]


def test_subset_stratification_is_reproducible_and_fills_shortage():
    records = []
    outcomes = {}
    for index in range(12):
        record = make_record(str(index))
        if index < 2:
            record["gold_packet_ids"] = [1]
        records.append(record); outcomes[str(index)] = index % 2 == 0
    first = choose_feasibility_records(records, outcomes, count=8, seed=20260803)
    second = choose_feasibility_records(records, outcomes, count=8, seed=20260803)
    assert [item["sample_id"] for item in first] == [item["sample_id"] for item in second]
    assert len(first) == 8
    assert sum(item["single_gold"] for item in first) == 2
