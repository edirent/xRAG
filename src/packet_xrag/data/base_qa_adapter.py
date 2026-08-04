"""Common interface and leakage-safe packetization for QA datasets."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod


PACKET_FORMAT = "[Title] sentence"
PACKETIZER_VERSION = "sentence-packets-v1"
MAX_PACKETS = 48


def normalize_answer_alias(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalized_question(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def stable_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def split_sentences(text):
    """Small deterministic sentence splitter without external model state."""
    text = " ".join(str(text).replace("\n", " ").split())
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'“‘(]*[A-Z0-9])", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def contains_alias(text, aliases):
    normalized = f" {normalize_answer_alias(text)} "
    return any(alias and f" {alias} " in normalized for alias in aliases)


class PacketQADatasetAdapter(ABC):
    dataset_name = "base"
    source_identifier = ""

    @abstractmethod
    def load_train(self):
        raise NotImplementedError

    @abstractmethod
    def load_validation(self):
        raise NotImplementedError

    def get_question(self, sample):
        return sample["question"]

    def get_answers(self, sample):
        answers = sample.get("answers") or [sample["answer"]]
        return list(dict.fromkeys(str(value) for value in answers if str(value).strip()))

    def get_documents(self, sample):
        return sample["documents"]

    def get_support_annotations(self, sample):
        return sample.get("support_annotations", [])

    @abstractmethod
    def canonicalize(self, sample):
        raise NotImplementedError

    def packetize(self, sample, max_packets=MAX_PACKETS):
        sample = self.canonicalize(sample) if "documents" not in sample else sample
        aliases = [normalize_answer_alias(value) for value in self.get_answers(sample)]
        packets = []
        for doc_index, document in enumerate(self.get_documents(sample)):
            title = " ".join(str(document.get("title") or f"Document {doc_index}").split())
            support = set(int(value) for value in document.get("support_sentence_ids", []))
            for sentence_index, sentence in enumerate(document["sentences"]):
                sentence = " ".join(str(sentence).split())
                if not sentence:
                    continue
                packets.append({"dataset": self.dataset_name, "packet_id": len(packets),
                    "document_id": str(document.get("document_id", doc_index)),
                    "doc_id": doc_index, "title": title, "sentence_id": sentence_index,
                    "sentence_text": sentence, "text": sentence,
                    "packet_text": f"[{title}] {sentence}",
                    "encoder_text": f"[{title}] {sentence}",
                    "is_support": sentence_index in support,
                    "is_supporting": sentence_index in support,
                    "contains_answer": contains_alias(sentence, aliases)})
                if len(packets) == max_packets:
                    return packets
        return packets

    def evaluate(self, predictions):
        from scripts.packet_xrag import run_selector_calibration as selector
        rows = []
        for item in predictions:
            scores = [selector.score_prediction(item["prediction"], answer)
                      for answer in item["answers"]]
            rows.append({"sample_id": item["sample_id"],
                         "em": max(value[0] for value in scores),
                         "f1": max(value[1] for value in scores)})
        return rows

    def inference_packets(self, packets):
        forbidden = {"is_support", "is_supporting", "contains_answer"}
        return [{key: value for key, value in packet.items() if key not in forbidden}
                for packet in packets]

