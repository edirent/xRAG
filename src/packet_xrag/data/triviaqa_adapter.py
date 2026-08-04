"""TriviaQA reading-comprehension adapter with answer-alias supervision."""

from datasets import load_dataset

from .base_qa_adapter import PacketQADatasetAdapter, split_sentences


class TriviaQAAdapter(PacketQADatasetAdapter):
    dataset_name = "triviaqa"
    source_identifier = "mandarjoshi/trivia_qa:rc"

    def load_train(self):
        return load_dataset("mandarjoshi/trivia_qa", "rc", split="train")

    def load_validation(self):
        return load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")

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

