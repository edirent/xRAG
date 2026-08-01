#!/usr/bin/env python
"""Evaluate frozen V1 xRAG projector packet representations on its locked validation split."""

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM

VARIANTS = ["V1_TITLE_SENTENCE", "V2_SENTENCE_ONLY", "V3_LOCAL_WINDOW_3", "V4_FORWARD_WINDOW_2", "V5_SUPPORT_DOCUMENT"]
REFERENCE_V1_F1 = 63.019047619047605
MAX_RETRIEVER_LENGTH = 180


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--projector-checkpoint", default="cache/projector/packet_projector_calibration/last/projector.pt")
    parser.add_argument("--training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--validation-ids-output", default="cache/results/packet_representation_ablation_validation_ids.json")
    parser.add_argument("--output", default="cache/results/packet_representation_ablation.jsonl")
    parser.add_argument("--summary-output", default="cache/results/packet_representation_ablation_summary.csv")
    parser.add_argument("--decision-output", default="cache/results/packet_representation_ablation_decision.md")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def fact_id(title, sentence_id):
    return f"{title}::{sentence_id}"


def make_chunk(variant, doc_id, chunk_id, title, encoder_text, covered_sentence_ids, gold):
    covered = [fact_id(title, sid) for sid in covered_sentence_ids if fact_id(title, sid) in gold]
    return {"variant": variant, "doc_id": doc_id, "chunk_id": chunk_id, "title": title, "encoder_text": encoder_text,
            "covered_sentence_ids": covered_sentence_ids, "covered_gold_fact_ids": covered}


def build_variant_chunks(sample, variant):
    titles = list(sample["context"]["title"])
    assert len(titles) == len(set(titles)), "context titles must map uniquely"
    gold = {fact_id(t, int(s)) for t, s in zip(sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"])}
    assert gold
    chunks = []
    for doc_id, (title, raw_sentences) in enumerate(zip(titles, sample["context"]["sentences"])):
        sentences = [sentence.strip() for sentence in raw_sentences]
        if variant == "V5_SUPPORT_DOCUMENT":
            covered = list(range(len(sentences)))
            if not any(fact_id(title, sid) in gold for sid in covered):
                continue
            text = " ".join(sentence for sentence in sentences if sentence)
            chunks.append(make_chunk(variant, doc_id, doc_id, title, f"[{title}]\n{text}", covered, gold))
            continue
        for sentence_id, sentence in enumerate(sentences):
            if not sentence:
                continue
            if variant == "V1_TITLE_SENTENCE":
                covered, text = [sentence_id], f"[{title}] {sentence}"
            elif variant == "V2_SENTENCE_ONLY":
                covered, text = [sentence_id], sentence
            elif variant == "V3_LOCAL_WINDOW_3":
                covered = [sid for sid in range(max(0, sentence_id - 1), min(len(sentences), sentence_id + 2)) if sentences[sid]]
                text = "\n".join([f"[{title}]"] + [sentences[sid] for sid in covered])
            elif variant == "V4_FORWARD_WINDOW_2":
                covered = [sid for sid in range(sentence_id, min(len(sentences), sentence_id + 2)) if sentences[sid]]
                text = "\n".join([f"[{title}]"] + [sentences[sid] for sid in covered])
            else:
                raise ValueError(variant)
            chunk = make_chunk(variant, doc_id, len(chunks), title, text, covered, gold)
            if chunk["covered_gold_fact_ids"]:
                chunks.append(chunk)
    assert chunks
    return chunks, gold


def exact_minimum_set_cover(chunks, gold):
    candidates = [i for i, chunk in enumerate(chunks) if chunk["covered_gold_fact_ids"]]
    token_costs = [len(chunk["encoder_text"].split()) for chunk in chunks]
    for size in range(1, len(candidates) + 1):
        valid = []
        for subset in itertools.combinations(candidates, size):
            coverage = set().union(*(set(chunks[i]["covered_gold_fact_ids"]) for i in subset))
            if coverage == gold:
                valid.append((sum(token_costs[i] for i in subset), subset))
        if valid:
            return list(min(valid)[1])
    raise AssertionError("gold facts are not coverable")


def load_locked_records(config_path, split_path, max_samples):
    config, split = json.loads(Path(config_path).read_text()), json.loads(Path(split_path).read_text())
    assert config["validation_sample_ids"] == split["validation_sample_ids"]
    ids = config["validation_sample_ids"][:max_samples]
    wanted, found = set(ids), {}
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", trust_remote_code=True)
    for sample in dataset:
        sid = str(sample.get("id", sample.get("_id")))
        if sid in wanted:
            found[sid] = sample
            if len(found) == len(wanted):
                break
    assert set(found) == wanted
    return [found[sid] for sid in ids], config["validation_sample_ids"]


def add_token_stats(chunks, tokenizer):
    for chunk in chunks:
        raw = tokenizer(chunk["encoder_text"], add_special_tokens=True, truncation=False)["input_ids"]
        used = tokenizer(chunk["encoder_text"], add_special_tokens=True, truncation=True, max_length=MAX_RETRIEVER_LENGTH)["input_ids"]
        chunk["raw_retriever_token_count"] = len(raw)
        chunk["used_retriever_token_count"] = len(used)
        chunk["was_truncated"] = len(raw) > len(used)


@torch.inference_mode()
def encode_chunks(chunks, tokenizer, model, device):
    tokenized = tokenizer([c["encoder_text"] for c in chunks], max_length=MAX_RETRIEVER_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
    embeddings = model.get_doc_embedding(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
    return embeddings.view(len(chunks), -1)


def substring_score(prediction, gold):
    pred, target = selector.normalize_answer(prediction), selector.normalize_answer(gold)
    return float(bool(pred) and (pred in target or target in pred))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = []
    v1_chunks = sum(r["num_selected_chunks"] for r in grouped[VARIANTS[0]]) / len(grouped[VARIANTS[0]])
    v1_f1 = 100 * sum(r["short_f1"] for r in grouped[VARIANTS[0]]) / len(grouped[VARIANTS[0]])
    for variant in VARIANTS:
        if variant not in grouped:
            continue
        items = grouped[variant]
        chunks = [chunk for row in items for chunk in row["selected_chunks"]]
        summary.append({"Variant": variant, "Short EM": 100 * sum(r["short_em"] for r in items) / len(items),
            "Short F1": 100 * sum(r["short_f1"] for r in items) / len(items), "Clean F1": 100 * sum(r["clean_f1"] for r in items) / len(items),
            "Substring": 100 * sum(r["substring_match"] for r in items) / len(items), "Avg chunks": sum(r["num_selected_chunks"] for r in items) / len(items),
            "Avg raw tokens/chunk": sum(c["raw_retriever_token_count"] for c in chunks) / len(chunks),
            "Avg used tokens/chunk": sum(c["used_retriever_token_count"] for c in chunks) / len(chunks),
            "Truncation rate": 100 * sum(c["was_truncated"] for c in chunks) / len(chunks),
            "F1 delta vs V1": 100 * sum(r["short_f1"] for r in items) / len(items) - v1_f1,
            "Avg chunks delta vs V1": sum(r["num_selected_chunks"] for r in items) / len(items) - v1_chunks})
    return summary


def choose_path(summary, reproducible):
    by = {row["Variant"]: row for row in summary}
    if not reproducible: return "Path E"
    assert set(by) == set(VARIANTS)
    if by["V2_SENTENCE_ONLY"]["Short F1"] - by["V1_TITLE_SENTENCE"]["Short F1"] >= 3: return "Path D"
    windows = max((by[v] for v in ["V3_LOCAL_WINDOW_3", "V4_FORWARD_WINDOW_2"]), key=lambda r: r["Short F1"])
    if windows["Short F1"] >= 66 and windows["F1 delta vs V1"] >= 3: return "Path A"
    best_v1_v4 = max(by[v]["Short F1"] for v in VARIANTS[:4])
    if by["V5_SUPPORT_DOCUMENT"]["Short F1"] - best_v1_v4 >= 5 and best_v1_v4 < 66: return "Path B"
    best = max(summary, key=lambda r: r["Short F1"])
    if all(r["Short F1"] < 66 for r in summary) and best["F1 delta vs V1"] < 3: return "Path C"
    return "No prescribed path"


def write_outputs(rows, summary, args, split_hash, reproducible):
    fields = list(summary[0])
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary_output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    best = max(summary, key=lambda row: row["Short F1"])
    path = choose_path(summary, reproducible)
    lines = ["# Packet Representation Ablation Decision", "", f"- Fixed validation split hash: `{split_hash}`",
        f"- V1 projector checkpoint: `{args.projector_checkpoint}`", f"- V1 reproduced within 1.0 F1: {'Yes' if reproducible else 'No'}",
        f"- V1 reproducibility gap: {summary[0]['Short F1'] - REFERENCE_V1_F1:.4f}",
        "- Historical 63.0190 reference scope: first 100 validation examples (`validation_generation_samples=100`)",
        "- Current requested scope: all 500 locked validation examples", "", "| Variant | Short EM | Short F1 | Clean F1 | Substring | Avg chunks | Truncation rate |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['Variant']} | {r['Short EM']:.2f} | {r['Short F1']:.2f} | {r['Clean F1']:.2f} | {r['Substring']:.2f} | {r['Avg chunks']:.3f} | {r['Truncation rate']:.2f}% |" for r in summary]
    lines += ["", f"- Best variant: {best['Variant']}", f"- Best F1 improvement vs V1: {best['F1 delta vs V1']:.4f}", f"- Final decision: {path}",
        "- Accessed final 100: No", "- Trained parameters: No", "- Started controller: No", ""]
    Path(args.decision_output).write_text("\n".join(lines))
    return path


def main():
    args = parse_args()
    assert 1 <= args.max_samples <= 500 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = torch.device(args.device); torch.cuda.set_device(device)
    records, all_validation_ids = load_locked_records(args.training_config, args.split_file, args.max_samples)
    split_hash = hashlib.sha256("".join(all_validation_ids).encode()).hexdigest()
    ids_record = {"num_validation_samples": 500, "sample_ids": all_validation_ids, "source": "existing V1 split", "split_hash": split_hash}
    Path(args.validation_ids_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.validation_ids_output).write_text(json.dumps(ids_record, indent=2) + "\n")

    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(device)
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN); model.set_xrag_token_id(xrag_token_id)
    model.projector.load_state_dict(torch.load(args.projector_checkpoint, map_location="cpu", weights_only=True), strict=True)
    for module in [sfr_model, model]:
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad = False
    assert not any(p.requires_grad for p in sfr_model.parameters()) and not any(p.requires_grad for p in model.parameters())

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w") as stream:
        for variant in VARIANTS:
            for sample_index, sample in enumerate(records):
                chunks, gold = build_variant_chunks(sample, variant)
                selected_indices = exact_minimum_set_cover(chunks, gold)
                selected = [chunks[i] for i in selected_indices]
                assert selected_indices and set().union(*(set(c["covered_gold_fact_ids"]) for c in selected)) == gold
                add_token_stats(selected, sfr_tokenizer)
                embeddings = encode_chunks(selected, sfr_tokenizer, sfr_model, device)
                assert embeddings.shape[0] == len(selected)
                raw, _, generated_tokens = selector.generate_xrag_answer(tokenizer, model, xrag_token_id, sample["question"], embeddings, device, args.max_new_tokens)
                clean, short = selector.clean_prediction(raw), selector.extract_short_answer(raw)
                if not short: short = "[EMPTY]"
                short_em, short_f1 = selector.score_prediction(short, sample["answer"])
                _, clean_f1 = selector.score_prediction(clean, sample["answer"])
                row = {"sample_id": str(sample.get("id", sample.get("_id"))), "variant": variant, "question": sample["question"], "gold_answer": sample["answer"],
                    "num_gold_facts": len(gold), "num_selected_chunks": len(selected), "selected_chunk_indices": selected_indices, "selected_chunks": selected,
                    "total_encoder_characters": sum(len(c["encoder_text"]) for c in selected), "total_raw_retriever_tokens": sum(c["raw_retriever_token_count"] for c in selected),
                    "total_used_retriever_tokens": sum(c["used_retriever_token_count"] for c in selected), "num_truncated_chunks": sum(c["was_truncated"] for c in selected),
                    "raw_prediction": raw, "clean_prediction": clean, "short_prediction": short, "short_em": short_em, "short_f1": short_f1,
                    "clean_f1": clean_f1, "substring_match": substring_score(short, sample["answer"]), "generated_tokens": generated_tokens}
                assert short_em in {0.0, 1.0} and 0 <= short_f1 <= 1 and 0 <= clean_f1 <= 1
                rows.append(row); stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
                if sample_index < 3:
                    print(json.dumps({"question": sample["question"], "gold_facts": sorted(gold), "variant": variant, "chunks": [c["encoder_text"] for c in selected], "prediction": short, "short_f1": short_f1}, ensure_ascii=False), flush=True)
            if variant == "V1_TITLE_SENTENCE" and args.max_samples == 500:
                current = 100 * sum(r["short_f1"] for r in rows) / 500
                if abs(current - REFERENCE_V1_F1) > 1.0:
                    summary = summarize(rows)
                    write_outputs(rows, summary, args, split_hash, False)
                    raise SystemExit(f"Path E: V1 F1 {current:.4f} does not reproduce {REFERENCE_V1_F1:.4f}")
    assert len(rows) == args.max_samples * 5
    summary = summarize(rows)
    path = write_outputs(rows, summary, args, split_hash, abs(summary[0]["Short F1"] - REFERENCE_V1_F1) <= 1.0)
    print(json.dumps({"rows": len(rows), "summary": summary, "path": path, "final_100_accessed": False, "training_run": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
