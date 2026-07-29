#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM


SFR_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
XRAG_MODEL_NAME = "Hannibal046/xrag-7b"
RANDOM_SEEDS = [13, 37, 73]


def parse_args():
    parser = argparse.ArgumentParser(description="Packet-xRAG representation and selector diagnostics.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default="cache/results/diagnostic_predictions.jsonl")
    parser.add_argument("--summary-output", default="cache/results/diagnostic_summary.csv")
    parser.add_argument("--decision-output", default="cache/results/diagnostic_decision.md")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def vram_gb(fn):
    return fn() / 1024**3


def print_vram(label):
    print(f"{label} allocated GB: {vram_gb(torch.cuda.memory_allocated):.3f}", flush=True)
    print(f"{label} reserved GB: {vram_gb(torch.cuda.memory_reserved):.3f}", flush=True)


def load_models(device, dtype):
    print(f"Device: {device}", flush=True)
    print(f"Requested dtype: {dtype}", flush=True)

    sfr_tokenizer = AutoTokenizer.from_pretrained(SFR_MODEL_NAME)
    sfr_model = SFR.from_pretrained(SFR_MODEL_NAME, torch_dtype=dtype).eval().to(device)
    print_vram("After SFR")
    print("SFR device:", next(sfr_model.parameters()).device, flush=True)
    print("SFR dtype:", next(sfr_model.parameters()).dtype, flush=True)
    print("SFR embedding dim:", sfr_model.get_embed_dim(), flush=True)
    assert sfr_model.get_embed_dim() == 4096

    xrag_tokenizer = AutoTokenizer.from_pretrained(XRAG_MODEL_NAME)
    config = AutoConfig.from_pretrained(XRAG_MODEL_NAME)
    xrag_model = XMistralForCausalLM.from_pretrained(
        XRAG_MODEL_NAME,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    assert XRAG_TOKEN in xrag_tokenizer.get_vocab(), f"{XRAG_TOKEN} missing from tokenizer"
    xrag_token_id = xrag_tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    xrag_model.set_xrag_token_id(xrag_token_id)

    print_vram("After both models")
    print("xRAG device:", next(xrag_model.parameters()).device, flush=True)
    print("xRAG dtype:", next(xrag_model.parameters()).dtype, flush=True)
    print("xRAG token:", XRAG_TOKEN, flush=True)
    print("xRAG token id:", xrag_token_id, flush=True)

    assert next(sfr_model.parameters()).device.type == "cuda"
    assert next(xrag_model.parameters()).device.type == "cuda"
    assert next(sfr_model.parameters()).device.index == 0
    assert next(xrag_model.parameters()).device.index == 0

    return sfr_tokenizer, sfr_model, xrag_tokenizer, xrag_model, xrag_token_id


def load_hotpotqa(max_samples):
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
    )
    return dataset.select(range(min(max_samples, len(dataset))))


def make_packets(sample):
    supporting_pairs = set(
        zip(
            sample["supporting_facts"]["title"],
            sample["supporting_facts"]["sent_id"],
        )
    )

    packets = []
    for doc_id, (title, sentences) in enumerate(
        zip(sample["context"]["title"], sample["context"]["sentences"])
    ):
        for sentence_id, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            packets.append(
                {
                    "doc_id": doc_id,
                    "packet_id": sentence_id,
                    "title": title,
                    "text": sentence,
                    "encoder_text": f"[{title}] {sentence}",
                    "is_supporting": (title, sentence_id) in supporting_pairs,
                }
            )

    assert packets
    assert any(packet["is_supporting"] for packet in packets)
    return packets


@torch.no_grad()
def encode_query_and_packets(tokenizer, model, question, packets, device):
    texts = [question] + [packet["encoder_text"] for packet in packets]
    tokenized = tokenizer(
        texts,
        max_length=180,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    embeddings = model.get_doc_embedding(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
    )
    embeddings = embeddings.view(-1, embeddings.shape[-1])
    assert embeddings.shape == (1 + len(packets), 4096), embeddings.shape

    query_raw = embeddings[0]
    packet_raw = embeddings[1:]
    query_norm = F.normalize(query_raw.float(), dim=-1)
    packet_norm = F.normalize(packet_raw.float(), dim=-1)
    return query_raw, packet_raw, query_norm, packet_norm


def select_oracle(packets):
    return [index for index, packet in enumerate(packets) if packet["is_supporting"]]


def select_topk(packet_norm, query_norm, budget):
    relevance = packet_norm @ query_norm
    selected = torch.topk(relevance, k=min(budget, packet_norm.shape[0])).indices.tolist()
    return sorted(selected)


def select_mmr(packet_norm, query_norm, budget, mmr_lambda):
    selected = []
    remaining = set(range(packet_norm.shape[0]))

    while remaining and len(selected) < budget:
        best_index = None
        best_score = float("-inf")
        for index in remaining:
            relevance_score = float(packet_norm[index] @ query_norm)
            if not selected:
                redundancy_score = 0.0
            else:
                redundancy_score = max(float(packet_norm[index] @ packet_norm[j]) for j in selected)
            score = mmr_lambda * relevance_score - (1.0 - mmr_lambda) * redundancy_score
            if score > best_score:
                best_index = index
                best_score = score
        selected.append(best_index)
        remaining.remove(best_index)

    return sorted(selected)


def deterministic_seed(sample_id, base_seed):
    digest = hashlib.sha256(f"{sample_id}:{base_seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def random_select(num_packets, budget, sample_id, base_seed):
    rng = random.Random(deterministic_seed(sample_id, base_seed))
    count = min(budget, num_packets)
    return sorted(rng.sample(range(num_packets), count))


def build_xrag_prompt(question, num_packets):
    background_tokens = " ".join([XRAG_TOKEN] * num_packets)
    return (
        "Refer to the background information and answer "
        "the question with a short answer.\n\n"
        f"Background: {background_tokens}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_text_oracle_prompt(question, packets, selected_indices):
    supporting_packets = sorted(
        [packets[index] for index in selected_indices],
        key=lambda packet: (packet["doc_id"], packet["packet_id"]),
    )
    background_text = "\n".join(
        f"[{packet['title']}] {packet['text']}" for packet in supporting_packets
    )
    prompt = (
        "Refer to the background information and answer "
        "the question with a short answer.\n\n"
        f"Background:\n{background_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    assert XRAG_TOKEN not in prompt
    return prompt


@torch.no_grad()
def generate_xrag_answer(
    tokenizer,
    model,
    xrag_token_id,
    question,
    selected_embeddings,
    device,
    max_new_tokens,
):
    prompt = build_xrag_prompt(question, selected_embeddings.shape[0])
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    num_xrag_tokens = (input_ids == xrag_token_id).sum().item()
    assert num_xrag_tokens == selected_embeddings.shape[0]

    generated_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        retrieval_embeds=selected_embeddings,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    prediction = tokenizer.batch_decode(generated_output, skip_special_tokens=True)[0]
    return prediction.strip()


@torch.no_grad()
def generate_text_answer(tokenizer, model, prompt, device, max_new_tokens):
    assert XRAG_TOKEN not in prompt
    retrieval_embeds = None
    assert retrieval_embeds is None
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    generated_output = model.generate(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = tokenized.input_ids.shape[1]
    prediction = tokenizer.batch_decode(
        generated_output[:, input_length:],
        skip_special_tokens=True,
    )[0]
    return prediction.strip()


def normalize_answer(text):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction, gold_answer):
    return exact_match_score(prediction, gold_answer), f1_score(prediction, gold_answer)


def coverage_metrics(selected_indices, gold_supporting_indices):
    selected = set(selected_indices)
    selected_supporting_count = len(selected & gold_supporting_indices)
    support_recall = selected_supporting_count / len(gold_supporting_indices)
    support_precision = selected_supporting_count / len(selected_indices) if selected_indices else 0.0
    full_support_coverage = float(gold_supporting_indices.issubset(selected))

    assert 0.0 <= support_recall <= 1.0
    assert 0.0 <= support_precision <= 1.0
    assert full_support_coverage in {0.0, 1.0}
    assert selected_supporting_count <= len(selected_indices)
    assert selected_supporting_count <= len(gold_supporting_indices)

    return selected_supporting_count, support_recall, support_precision, full_support_coverage


def method_runs(sample_id, packets, packet_norm, query_norm, mmr_lambda):
    oracle_indices = sorted(select_oracle(packets))
    yield "TEXT_ORACLE", None, None, oracle_indices, "text"
    yield "XRAG_ORACLE", None, None, oracle_indices, "xrag"
    yield "ORACLE_1", 1, None, oracle_indices[:1], "xrag"
    yield "ORACLE_2", 2, None, oracle_indices[:2], "xrag"

    for budget in [2, 4]:
        for base_seed in RANDOM_SEEDS:
            yield f"RANDOM_{budget}", budget, base_seed, random_select(len(packets), budget, sample_id, base_seed), "xrag"

    for budget in [2, 4]:
        yield f"TOPK_{budget}", budget, None, select_topk(packet_norm, query_norm, budget), "xrag"

    for budget in [2, 4]:
        yield f"MMR_{budget}", budget, None, select_mmr(packet_norm, query_norm, budget, mmr_lambda), "xrag"


def summary_method_budget(row):
    method = row["method"]
    if method.startswith("RANDOM_"):
        return "RANDOM", row["budget"]
    if method.startswith("TOPK_"):
        return "TOPK", row["budget"]
    if method.startswith("MMR_"):
        return "MMR", row["budget"]
    return method, row["budget"]


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def write_summary(predictions, summary_output):
    groups = defaultdict(list)
    for row in predictions:
        groups[summary_method_budget(row)].append(row)

    ordered_keys = [
        ("TEXT_ORACLE", None),
        ("XRAG_ORACLE", None),
        ("ORACLE_1", 1),
        ("ORACLE_2", 2),
        ("RANDOM", 2),
        ("RANDOM", 4),
        ("TOPK", 2),
        ("TOPK", 4),
        ("MMR", 2),
        ("MMR", 4),
    ]

    rows = []
    for method, budget in ordered_keys:
        items = groups[(method, budget)]
        assert items, (method, budget)
        rows.append(
            {
                "Method": method,
                "Budget": "" if budget is None else budget,
                "EM": mean([item["em"] for item in items]),
                "F1": mean([item["f1"] for item in items]),
                "Avg packets": mean([item["num_packets"] for item in items]),
                "Packet ratio": mean([item["packet_ratio"] for item in items]),
                "Support recall": mean([item["support_recall"] for item in items]),
                "Support precision": mean([item["support_precision"] for item in items]),
                "Full-support coverage": mean([item["full_support_coverage"] for item in items]),
                "F1 std": std([item["f1"] for item in items]),
                "Support recall std": std([item["support_recall"] for item in items]),
            }
        )

    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method",
        "Budget",
        "EM",
        "F1",
        "Avg packets",
        "Packet ratio",
        "Support recall",
        "Support precision",
        "Full-support coverage",
        "F1 std",
        "Support recall std",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary", flush=True)
    print(
        "| Method | Budget | EM | F1 | Avg packets | Packet ratio | "
        "Support recall | Support precision | Full-support coverage |",
        flush=True,
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for row in rows:
        print(
            f"| {row['Method']} | {row['Budget']} | "
            f"{100 * row['EM']:.2f} | {100 * row['F1']:.2f} | "
            f"{row['Avg packets']:.2f} | {row['Packet ratio']:.3f} | "
            f"{row['Support recall']:.3f} | {row['Support precision']:.3f} | "
            f"{row['Full-support coverage']:.3f} |",
            flush=True,
        )
    return rows


def find_summary_row(rows, method, budget=None):
    for row in rows:
        if row["Method"] == method and row["Budget"] == ("" if budget is None else budget):
            return row
    raise KeyError((method, budget))


def build_decision(rows):
    text_oracle = find_summary_row(rows, "TEXT_ORACLE")
    xrag_oracle = find_summary_row(rows, "XRAG_ORACLE")
    oracle_1 = find_summary_row(rows, "ORACLE_1", 1)
    oracle_2 = find_summary_row(rows, "ORACLE_2", 2)
    random_2 = find_summary_row(rows, "RANDOM", 2)
    random_4 = find_summary_row(rows, "RANDOM", 4)
    topk_2 = find_summary_row(rows, "TOPK", 2)
    topk_4 = find_summary_row(rows, "TOPK", 4)
    mmr_2 = find_summary_row(rows, "MMR", 2)
    mmr_4 = find_summary_row(rows, "MMR", 4)

    representation_gap = text_oracle["F1"] - xrag_oracle["F1"]
    selector_high_recall_low_f1 = (
        max(topk_2["Support recall"], topk_4["Support recall"], mmr_2["Support recall"], mmr_4["Support recall"]) >= 0.75
        and max(topk_2["F1"], topk_4["F1"], mmr_2["F1"], mmr_4["F1"]) <= xrag_oracle["F1"] - 0.05
    )
    path_a = representation_gap >= 0.15 or selector_high_recall_low_f1
    path_b = (
        representation_gap <= 0.10
        and mmr_2["Support recall"] < xrag_oracle["Support recall"]
        and mmr_2["F1"] - random_2["F1"] >= 0.05
        and xrag_oracle["F1"] - mmr_2["F1"] >= 0.05
    )
    path_c = text_oracle["F1"] < 0.50
    path_d = (
        abs(random_2["F1"] - topk_2["F1"]) < 0.03
        and abs(random_2["F1"] - mmr_2["F1"]) < 0.03
    )

    if path_c:
        final_path = "Path C: fix prompt, answer extraction, or evaluation before training."
    elif path_a:
        final_path = "Path A: train packet-level projector."
    elif path_b:
        final_path = "Path B: train adaptive selector/controller."
    elif path_d:
        final_path = "Path D: data shortcut risk; build controlled redundancy/counterfactual distractors."
    else:
        final_path = "Path C: inspect prompt/evaluation because diagnostics are not clean enough for training."

    lines = [
        "# Packet-xRAG Diagnostic Decision",
        "",
        f"1. TEXT_ORACLE F1: {100 * text_oracle['F1']:.2f}",
        f"2. XRAG_ORACLE F1: {100 * xrag_oracle['F1']:.2f}",
        f"3. Representation gap: {100 * representation_gap:.2f}",
        "",
        "4. RANDOM/TOPK/MMR comparison:",
        f"- RANDOM_2 F1: {100 * random_2['F1']:.2f}",
        f"- TOPK_2 F1: {100 * topk_2['F1']:.2f}",
        f"- MMR_2 F1: {100 * mmr_2['F1']:.2f}",
        f"- RANDOM_4 F1: {100 * random_4['F1']:.2f}",
        f"- TOPK_4 F1: {100 * topk_4['F1']:.2f}",
        f"- MMR_4 F1: {100 * mmr_4['F1']:.2f}",
        "",
        "5. Support recall and full-support coverage:",
        f"- TOPK_2 support recall: {topk_2['Support recall']:.3f}",
        f"- MMR_2 support recall: {mmr_2['Support recall']:.3f}",
        f"- TOPK_2 full-support coverage: {topk_2['Full-support coverage']:.3f}",
        f"- MMR_2 full-support coverage: {mmr_2['Full-support coverage']:.3f}",
        "",
        "6. ORACLE_1/2 results:",
        f"- ORACLE_1 F1: {100 * oracle_1['F1']:.2f}",
        f"- ORACLE_2 F1: {100 * oracle_2['F1']:.2f}",
        f"- XRAG_ORACLE F1: {100 * xrag_oracle['F1']:.2f}",
        "",
        f"7. Final choice: {final_path}",
        "",
        "Notes:",
        f"- Path A trigger: {path_a}",
        f"- Path B trigger: {path_b}",
        f"- Path C trigger: {path_c}",
        f"- Path D trigger: {path_d}",
    ]

    comparisons = {
        "text_oracle_f1": text_oracle["F1"],
        "xrag_oracle_f1": xrag_oracle["F1"],
        "representation_gap": representation_gap,
        "oracle_1_f1": oracle_1["F1"],
        "oracle_2_f1": oracle_2["F1"],
        "random_2_f1": random_2["F1"],
        "topk_2_f1": topk_2["F1"],
        "mmr_2_f1": mmr_2["F1"],
        "random_4_f1": random_4["F1"],
        "topk_4_f1": topk_4["F1"],
        "mmr_4_f1": mmr_4["F1"],
        "topk_2_support_recall": topk_2["Support recall"],
        "mmr_2_support_recall": mmr_2["Support recall"],
        "topk_2_full_support_coverage": topk_2["Full-support coverage"],
        "mmr_2_full_support_coverage": mmr_2["Full-support coverage"],
        "final_path": final_path,
    }
    return "\n".join(lines) + "\n", comparisons


def print_required_comparisons(comparisons):
    print("", flush=True)
    print(f"TEXT_ORACLE F1: {100 * comparisons['text_oracle_f1']:.2f}", flush=True)
    print(f"XRAG_ORACLE F1: {100 * comparisons['xrag_oracle_f1']:.2f}", flush=True)
    print(f"TEXT-XRAG oracle gap: {100 * comparisons['representation_gap']:.2f}", flush=True)
    print("", flush=True)
    print(f"ORACLE_1 F1: {100 * comparisons['oracle_1_f1']:.2f}", flush=True)
    print(f"ORACLE_2 F1: {100 * comparisons['oracle_2_f1']:.2f}", flush=True)
    print(f"XRAG_ORACLE F1: {100 * comparisons['xrag_oracle_f1']:.2f}", flush=True)
    print("", flush=True)
    print(f"RANDOM_2 F1: {100 * comparisons['random_2_f1']:.2f}", flush=True)
    print(f"TOPK_2 F1: {100 * comparisons['topk_2_f1']:.2f}", flush=True)
    print(f"MMR_2 F1: {100 * comparisons['mmr_2_f1']:.2f}", flush=True)
    print("", flush=True)
    print(f"RANDOM_4 F1: {100 * comparisons['random_4_f1']:.2f}", flush=True)
    print(f"TOPK_4 F1: {100 * comparisons['topk_4_f1']:.2f}", flush=True)
    print(f"MMR_4 F1: {100 * comparisons['mmr_4_f1']:.2f}", flush=True)
    print("", flush=True)
    print(f"TOPK_2 support recall: {comparisons['topk_2_support_recall']:.3f}", flush=True)
    print(f"MMR_2 support recall: {comparisons['mmr_2_support_recall']:.3f}", flush=True)
    print(f"TOPK_2 full-support coverage: {comparisons['topk_2_full_support_coverage']:.3f}", flush=True)
    print(f"MMR_2 full-support coverage: {comparisons['mmr_2_full_support_coverage']:.3f}", flush=True)
    print(f"Decision: {comparisons['final_path']}", flush=True)


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    sfr_tokenizer, sfr_model, xrag_tokenizer, xrag_model, xrag_token_id = load_models(device, dtype)

    print(f"Loading HotpotQA distractor validation max_samples={args.max_samples}", flush=True)
    dataset = load_hotpotqa(args.max_samples)
    print(f"Loaded samples: {len(dataset)}", flush=True)

    predictions = []
    method_counts = defaultdict(int)
    seed_counts = defaultdict(int)
    start_all = time.perf_counter()
    with output_path.open("w") as out:
        for sample_index, sample in enumerate(dataset):
            sample_id = str(sample.get("id", sample.get("_id", sample_index)))
            question = sample["question"]
            gold_answer = sample["answer"]
            packets = make_packets(sample)
            gold_supporting_indices = {
                index for index, packet in enumerate(packets) if packet["is_supporting"]
            }

            encode_start = time.perf_counter()
            _, packet_raw, query_norm, packet_norm = encode_query_and_packets(
                sfr_tokenizer,
                sfr_model,
                question,
                packets,
                device,
            )
            encode_seconds = time.perf_counter() - encode_start

            print(
                f"[{sample_index + 1}/{len(dataset)}] {sample_id} "
                f"packets={len(packets)} oracle={len(gold_supporting_indices)} "
                f"encode={encode_seconds:.3f}s",
                flush=True,
            )

            per_sample_methods = []
            for method, budget, random_seed, selected_indices, input_mode in method_runs(
                sample_id,
                packets,
                packet_norm,
                query_norm,
                args.mmr_lambda,
            ):
                assert selected_indices
                selected_supporting_count, support_recall, support_precision, full_support_coverage = coverage_metrics(
                    selected_indices,
                    gold_supporting_indices,
                )

                generate_start = time.perf_counter()
                if input_mode == "text":
                    prompt = build_text_oracle_prompt(question, packets, selected_indices)
                    prediction = generate_text_answer(
                        xrag_tokenizer,
                        xrag_model,
                        prompt,
                        device,
                        args.max_new_tokens,
                    )
                else:
                    selected_embeddings = packet_raw[selected_indices]
                    assert selected_embeddings.shape[0] == len(selected_indices)
                    prediction = generate_xrag_answer(
                        xrag_tokenizer,
                        xrag_model,
                        xrag_token_id,
                        question,
                        selected_embeddings,
                        device,
                        args.max_new_tokens,
                    )
                generate_seconds = time.perf_counter() - generate_start
                assert prediction.strip()

                em, f1 = score_prediction(prediction, gold_answer)
                row = {
                    "sample_id": sample_id,
                    "method": method,
                    "budget": budget,
                    "random_seed": random_seed,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "num_packets": len(selected_indices),
                    "total_packets": len(packets),
                    "packet_ratio": len(selected_indices) / len(packets),
                    "selected_indices": selected_indices,
                    "gold_supporting_indices": sorted(gold_supporting_indices),
                    "selected_supporting_count": selected_supporting_count,
                    "support_recall": support_recall,
                    "support_precision": support_precision,
                    "full_support_coverage": full_support_coverage,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                predictions.append(row)
                method_counts[(sample_id, method)] += 1
                if random_seed is not None:
                    seed_counts[(sample_id, method)] += 1
                per_sample_methods.append(method)

                print(
                    f"  {method} seed={random_seed} n={len(selected_indices)} "
                    f"rec={support_recall:.2f} full={full_support_coverage:.0f} "
                    f"em={em:.0f} f1={f1:.3f} gen={generate_seconds:.3f}s "
                    f"pred={prediction[:80]!r}",
                    flush=True,
                )

            required_once = {"TEXT_ORACLE", "XRAG_ORACLE", "ORACLE_1", "ORACLE_2", "TOPK_2", "TOPK_4", "MMR_2", "MMR_4"}
            assert required_once.issubset(set(per_sample_methods))
            assert per_sample_methods.count("RANDOM_2") == 3
            assert per_sample_methods.count("RANDOM_4") == 3

    elapsed = time.perf_counter() - start_all
    print(f"\nTotal runtime seconds: {elapsed:.3f}", flush=True)
    print(f"Peak allocated GB: {vram_gb(torch.cuda.max_memory_allocated):.3f}", flush=True)
    print(f"Peak reserved GB: {vram_gb(torch.cuda.max_memory_reserved):.3f}", flush=True)
    print(f"Predictions written to: {output_path}", flush=True)

    expected_rows = len(dataset) * 14
    assert len(predictions) == expected_rows, (len(predictions), expected_rows)

    summary_rows = write_summary(predictions, args.summary_output)
    print(f"Summary written to: {args.summary_output}", flush=True)

    decision_md, comparisons = build_decision(summary_rows)
    decision_path = Path(args.decision_output)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(decision_md)
    print_required_comparisons(comparisons)
    print(f"Decision written to: {decision_path}", flush=True)


if __name__ == "__main__":
    main()
