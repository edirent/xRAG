"""2WikiMultiHopQA adapter."""

from datasets import load_dataset

from .base_qa_adapter import PacketQADatasetAdapter


class TwoWikiAdapter(PacketQADatasetAdapter):
    dataset_name = "2wiki"
    source_identifier = "framolfese/2WikiMultihopQA"

    def load_train(self):
        return load_dataset(self.source_identifier, split="train")

    def load_validation(self):
        return load_dataset(self.source_identifier, split="validation")

    def canonicalize(self, sample):
        titles = list(sample["context"]["title"])
        sentence_groups = list(sample["context"]["sentences"])
        support_pairs = set(zip(sample["supporting_facts"]["title"],
                                sample["supporting_facts"]["sent_id"]))
        documents = [{"document_id": index, "title": title, "sentences": list(sentences),
                      "support_sentence_ids": [sid for sid in range(len(sentences))
                                               if (title, sid) in support_pairs]}
                     for index, (title, sentences) in enumerate(zip(titles, sentence_groups))]
        return {"id": str(sample["id"]), "question": sample["question"],
                "answer": sample["answer"], "answers": [sample["answer"]],
                "documents": documents, "support_annotations": sorted(support_pairs),
                "metadata": {"type": sample.get("type", "unknown"),
                             "hop_count": len(sample.get("evidences", []))}}

