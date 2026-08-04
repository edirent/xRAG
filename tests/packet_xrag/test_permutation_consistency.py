from src.packet_xrag.composition.diagnostics import document_order, fixed_permutation


def test_fixed_permutation_is_reproducible_and_set_preserving():
    values = [1, 2, 3, 4, 5, 6]
    first = fixed_permutation(values, "sample", 0)
    assert first == fixed_permutation(values, "sample", 0)
    assert sorted(first) == values


def test_document_order_is_deterministic():
    record = {"packets": [{"doc_id": 1, "sentence_id": 0},
                           {"doc_id": 0, "sentence_id": 2},
                           {"doc_id": 0, "sentence_id": 1}]}
    assert document_order(record, [0, 1, 2]) == [2, 1, 0]
