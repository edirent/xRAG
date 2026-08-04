import json

from src.packet_xrag.data.base_qa_adapter import (
    PacketQADatasetAdapter, normalize_answer_alias,
)
from src.packet_xrag.data.musique_adapter import MusiqueAdapter
from src.packet_xrag.data.twowiki_adapter import TwoWikiAdapter


def test_answer_alias_normalization_is_deterministic():
    assert normalize_answer_alias("The École, A.") == "école"
    assert normalize_answer_alias("The École, A.") == normalize_answer_alias("the école a")


def test_twowiki_packetization_and_support_labels_are_deterministic():
    sample = {"id": "x", "question": "q?", "answer": "Alpha", "type": "bridge",
        "evidences": [["T", "r", "x"]],
        "supporting_facts": {"title": ["T"], "sent_id": [1]},
        "context": {"title": ["T"], "sentences": [["First.", "Alpha here."]]}}
    adapter = TwoWikiAdapter(); canonical = adapter.canonicalize(sample)
    first = adapter.packetize(canonical); second = adapter.packetize(canonical)
    assert first == second
    assert first[1]["packet_text"] == "[T] Alpha here."
    assert first[1]["is_support"] and first[1]["contains_answer"]


def test_musique_paragraph_packetization_is_stable():
    sample = {"id": "2hop__x", "question": "q?", "answer": "Beta",
        "answer_aliases": ["B"], "answerable": True, "question_decomposition": [{}, {}],
        "paragraphs": [{"idx": 3, "title": "Doc", "paragraph_text":
                         "First sentence. Beta follows!", "is_supporting": True}]}
    adapter = MusiqueAdapter(); packets = adapter.packetize(adapter.canonicalize(sample))
    assert [packet["sentence_id"] for packet in packets] == [0, 1]
    assert all(packet["is_support"] for packet in packets)


def test_inference_packet_serialization_removes_supervision():
    adapter = TwoWikiAdapter()
    packets = [{"packet_text": "[T] x", "is_support": True,
                "is_supporting": True, "contains_answer": True}]
    view = adapter.inference_packets(packets)[0]
    assert view == {"packet_text": "[T] x"}
