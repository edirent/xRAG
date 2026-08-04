"""TriviaQA reading-comprehension adapter with answer-alias supervision."""

from datasets import load_dataset

from .base_qa_adapter import PacketQADatasetAdapter, split_sentences


class TriviaQAAdapter(PacketQADatasetAdapter):
    dataset_name = "triviaqa"
    source_identifier = "mandarjoshi/trivia_qa:rc"

    def load_train(self):
        return self._load_unique("train")

    def load_validation(self):
        return self._load_unique("validation")

    def _load_unique(self, split):
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
        # The RC configuration contains two evidence variants for many question
        # IDs. Freeze the richer supplied-evidence variant without using answers.
        best = {}
        for index, sample in enumerate(dataset):
            contexts = [*sample["entity_pages"]["wiki_context"],
                        *sample["search_results"]["search_context"]]
            score = (sum(bool(str(value).strip()) for value in contexts),
                     sum(len(str(value)) for value in contexts), -index)
            sid = str(sample["question_id"])
            if sid not in best or score > best[sid][0]: best[sid] = (score, index)
        return dataset.select(sorted(value[1] for value in best.values()))

    def canonicalize(self, sample):
        documents = []
        entity = sample["entity_pages"]
        for index, (title, context) in enumerate(zip(entity["title"], entity["wiki_context"])):
            documents.append({"document_id": f"entity:{index}", "title": title,
                              "sentences": split_sentences(context), "support_sentence_ids": []})
        search = sample["search_results"]
        for index, (title, context) in enumerate(zip(search["title"], search["search_context"])):
            documents.append({"document_id": f"search:{index}", "title": title,
                              "sentences": split_sentences(context), "support_sentence_ids": []})
        answers = [sample["answer"]["value"], *sample["answer"]["aliases"]]
        return {"id": str(sample["question_id"]), "question": sample["question"],
                "answer": sample["answer"]["value"], "answers": answers,
                "documents": documents, "support_annotations": [],
                "metadata": {"question_source": sample.get("question_source", "")}}
