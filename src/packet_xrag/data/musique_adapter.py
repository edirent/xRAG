"""MuSiQue answerable adapter."""

from datasets import load_dataset

from .base_qa_adapter import PacketQADatasetAdapter, split_sentences


class MusiqueAdapter(PacketQADatasetAdapter):
    dataset_name = "musique"
    source_identifier = "bdsaglam/musique"

    def load_train(self):
        return self._load_answerable("train")

    def load_validation(self):
        return self._load_answerable("validation")

    def _load_answerable(self, split):
        # This mirror concatenates the answerable and unanswerable variants under
        # the same IDs.  The protocol freezes the original answerable task only.
        return load_dataset(self.source_identifier, split=split).filter(
            lambda sample: bool(sample["answerable"]),
            load_from_cache_file=True,
            desc=f"Selecting MuSiQue answerable {split}",
        )

    def canonicalize(self, sample):
        documents = []
        for paragraph in sample["paragraphs"]:
            sentences = split_sentences(paragraph["paragraph_text"])
            documents.append({"document_id": paragraph["idx"], "title": paragraph["title"],
                              "sentences": sentences,
                              "support_sentence_ids": (list(range(len(sentences)))
                                                       if paragraph["is_supporting"] else [])})
        aliases = [sample["answer"], *sample.get("answer_aliases", [])]
        hop_count = len(sample.get("question_decomposition", []))
        if not hop_count:
            try: hop_count = int(str(sample["id"]).split("hop", 1)[0])
            except ValueError: hop_count = 0
        return {"id": str(sample["id"]), "question": sample["question"],
                "answer": sample["answer"], "answers": aliases, "documents": documents,
                "support_annotations": [p["idx"] for p in sample["paragraphs"]
                                        if p["is_supporting"]],
                "metadata": {"hop_count": hop_count, "answerable": sample.get("answerable", True)}}
