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
    parser = argparse.ArgumentParser(description="Packet-xRAG selector calibration.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default="cache/results/selector_calibration.jsonl")
    parser.add_argument("--summary-output", default="cache/results/selector_calibration_summary.csv")
    parser.add_argument("--decision-output", default="cache/results/selector_calibration_decision.md")
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

    xrag_tokenizer = AutoTokenizer.from_pretrained(
        XRAG_MODEL_NAME,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
    )
    if xrag_tokenizer.pad_token_id is None:
        if xrag_tokenizer.unk_token_id is not None:
            xrag_tokenizer.pad_token_id = xrag_tokenizer.unk_token_id
        else:
            xrag_tokenizer.pad_token_id = xrag_tokenizer.eos_token_id
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
    content = (
        "Refer to the background document and answer the question. "
        "Respond only with the shortest possible answer. "
        "Do not provide an explanation."
        "\n\n"
        f"Background: {background_tokens}"
        "\n\n"
        f"Question: {question}"
    )
    return f"[INST] {content} [/INST] The answer is:"


def build_text_oracle_prompt(question, packets, selected_indices):
    supporting_packets = sorted(
        [packets[index] for index in selected_indices],
        key=lambda packet: (packet["doc_id"], packet["packet_id"]),
    )
    background_text = "\n".join(
        f"[{packet['title']}] {packet['text']}" for packet in supporting_packets
    )
    content = (
        "Refer to the background document and answer the question. "
        "Respond only with the shortest possible answer. "
        "Do not provide an explanation."
        "\n\n"
        f"Background: {background_text}"
        "\n\n"
        f"Question: {question}"
    )
    prompt = f"[INST] {content} [/INST] The answer is:"
    assert XRAG_TOKEN not in prompt
    return prompt


def build_no_context_prompt(question):
    return (
        "[INST] Answer the question with the shortest possible answer.\n"
        "Do not provide an explanation.\n\n"
        f"Question: {question}\n"
        "[/INST] The answer is:"
    )


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
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
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
        pad_token_id=tokenizer.pad_token_id,
    )
    if generated_output.shape[1] > input_ids.shape[1]:
        new_tokens = generated_output[:, input_ids.shape[1] :]
    else:
        # XMistral's retrieval-aware generation path may return only newly
        # generated token ids rather than prompt + continuation.
        new_tokens = generated_output
    raw_prediction = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
    assert raw_prediction.strip(), (
        generated_output.shape,
        input_ids.shape,
        new_tokens.tolist(),
    )
    return raw_prediction, input_ids.shape[1], new_tokens.shape[1]


@torch.no_grad()
def generate_text_answer(tokenizer, model, prompt, device, max_new_tokens):
    assert XRAG_TOKEN not in prompt
    retrieval_embeds = None
    assert retrieval_embeds is None
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    generated_output = model.generate(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    input_length = tokenized.input_ids.shape[1]
    raw_prediction = tokenizer.batch_decode(
        generated_output[:, input_length:],
        skip_special_tokens=False,
    )[0]
    assert raw_prediction.strip()
    return raw_prediction, input_length, generated_output.shape[1] - input_length


def clean_prediction(text):
    text = text.replace("</s>", " ").replace("<s>", " ").strip()
    for marker in ["\nQuestion:", "\nBackground:", "\n[INST]", "\n[/INST]"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    cleaned = " ".join(text.split()).strip()
    # A retrieval-conditioned decode can occasionally emit only BOS/EOS.
    # Keep that model failure explicit and scoreable instead of aborting the run.
    return cleaned or "[EMPTY]"


def extract_short_answer(text):
    first_line = clean_prediction(text).splitlines()[0].strip()
    lower = first_line.lower()
    for prefix in ["the answer is ", "answer: ", "it is "]:
        if lower.startswith(prefix):
            first_line = first_line[len(prefix) :].strip()
            break
    if "." in first_line:
        candidate = first_line.split(".", 1)[0].strip()
        if candidate:
            first_line = candidate
    return first_line.strip(" \t\n\"'")


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
    yield "NO_CONTEXT", 0, None, [], "no_context"
    yield "TEXT_ORACLE", None, None, oracle_indices, "text"
    yield "XRAG_ORACLE", None, None, oracle_indices, "xrag"
    yield "ORACLE_1", 1, None, oracle_indices[:1], "xrag"
    yield "ORACLE_2", 2, None, oracle_indices[:2], "xrag"
    yield "ALL", None, None, list(range(len(packets))), "xrag"

    for budget in [2, 4]:
        for base_seed in RANDOM_SEEDS:
            yield f"RANDOM_{budget}", budget, base_seed, random_select(len(packets), budget, sample_id, base_seed), "xrag"

    for budget in [1, 2, 3, 4]:
        yield f"TOPK_{budget}", budget, None, select_topk(packet_norm, query_norm, budget), "xrag"

    for budget in [1, 2, 3, 4]:
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


def write_summary(predictions, summary_output):
    groups = defaultdict(list)
    for row in predictions:
        groups[summary_method_budget(row)].append(row)

    ordered_keys = [
        ("NO_CONTEXT", 0),
        ("TEXT_ORACLE", None),
        ("XRAG_ORACLE", None),
        ("ORACLE_1", 1),
        ("ORACLE_2", 2),
        ("ALL", None),
        ("RANDOM", 2),
        ("RANDOM", 4),
        ("TOPK", 1),
        ("TOPK", 2),
        ("TOPK", 3),
        ("TOPK", 4),
        ("MMR", 1),
        ("MMR", 2),
        ("MMR", 3),
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
                "Short EM": 100 * mean([item["short_em"] for item in items]),
                "Short F1": 100 * mean([item["short_f1"] for item in items]),
                "Clean F1": 100 * mean([item["clean_f1"] for item in items]),
                "Avg packets": mean([item["num_packets"] for item in items]),
                "Support recall": mean([item["support_recall"] for item in items]),
                "Full support": mean([item["full_support_coverage"] for item in items]),
            }
        )

    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method",
        "Budget",
        "Short EM",
        "Short F1",
        "Clean F1",
        "Avg packets",
        "Support recall",
        "Full support",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary", flush=True)
    print(
        "| Method | Budget | Short EM | Short F1 | Clean F1 | Avg packets | "
        "Support recall | Full support |",
        flush=True,
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for row in rows:
        print(
            f"| {row['Method']} | {row['Budget']} | "
            f"{row['Short EM']:.2f} | {row['Short F1']:.2f} | {row['Clean F1']:.2f} | "
            f"{row['Avg packets']:.2f} | {row['Support recall']:.3f} | "
            f"{row['Full support']:.3f} |",
            flush=True,
        )
    return rows


def find_summary_row(rows, method, budget=None):
    for row in rows:
        if row["Method"] == method and row["Budget"] == ("" if budget is None else budget):
            return row
    raise KeyError((method, budget))


def build_decision(rows):
    no_context = find_summary_row(rows, "NO_CONTEXT", 0)
    text_oracle = find_summary_row(rows, "TEXT_ORACLE")
    xrag_oracle = find_summary_row(rows, "XRAG_ORACLE")
    oracle_1 = find_summary_row(rows, "ORACLE_1", 1)
    oracle_2 = find_summary_row(rows, "ORACLE_2", 2)
    all_packets = find_summary_row(rows, "ALL")
    random_2 = find_summary_row(rows, "RANDOM", 2)
    random_4 = find_summary_row(rows, "RANDOM", 4)
    heuristic_rows = [
        find_summary_row(rows, method, budget)
        for method in ("TOPK", "MMR")
        for budget in (1, 2, 3, 4)
    ]
    best_heuristic = max(heuristic_rows, key=lambda row: row["Short F1"])

    text_no_context = text_oracle["Short F1"] - no_context["Short F1"]
    representation_gap = text_oracle["Short F1"] - xrag_oracle["Short F1"]
    oracle_heuristic_gap = xrag_oracle["Short F1"] - best_heuristic["Short F1"]

    random_by_budget = {2: random_2, 4: random_4}
    query_aware_gaps = []
    for row in heuristic_rows:
        budget = row["Budget"]
        if budget in random_by_budget:
            query_aware_gaps.append(
                (row["Short F1"] - random_by_budget[budget]["Short F1"], row)
            )
    best_query_gap, best_query_row = max(query_aware_gaps, key=lambda pair: pair[0])
    mmr_topk_gaps = {
        budget: find_summary_row(rows, "MMR", budget)["Short F1"]
        - find_summary_row(rows, "TOPK", budget)["Short F1"]
        for budget in (1, 2, 3, 4)
    }

    representation_pass = representation_gap <= 8.0
    projector_needed = representation_gap > 15.0
    selection_signal = best_query_gap >= 5.0
    controller_space = (
        oracle_heuristic_gap >= 5.0
        and best_heuristic["Support recall"] < xrag_oracle["Support recall"]
        and best_heuristic["Full support"] < xrag_oracle["Full support"]
    )
    all_collapse = best_heuristic["Short F1"] - all_packets["Short F1"] >= 5.0
    best_budget = best_heuristic["Budget"]
    stop_signal = best_budget in (2, 3) and all_collapse
    multihop_gain = oracle_2["Short F1"] - oracle_1["Short F1"]
    if projector_needed:
        final_recommendation = (
            "Train/calibrate the packet projector first. Preserve the k=1..4 "
            "selector benchmark for reevaluation; do not train the controller yet."
        )
    elif controller_space:
        final_recommendation = (
            "Proceed to a controller with an explicit STOP action."
            if stop_signal
            else "Proceed to controller calibration."
        )
    elif selection_signal:
        final_recommendation = (
            "Selection is useful, but the oracle gap is not large enough to "
            "justify controller training yet."
        )
    else:
        final_recommendation = (
            "Do not train a controller; strengthen controlled selector examples first."
        )

    lines = [
        "# Packet-xRAG Selector Calibration Decision",
        "",
        "## Required comparisons",
        "",
        f"- TEXT_ORACLE - NO_CONTEXT: {text_no_context:.2f}",
        f"- TEXT_ORACLE - XRAG_ORACLE: {representation_gap:.2f}",
        f"- XRAG_ORACLE - best heuristic: {oracle_heuristic_gap:.2f}",
        f"- Best heuristic: {best_heuristic['Method']}_{best_heuristic['Budget']} "
        f"({best_heuristic['Short F1']:.2f})",
        f"- Best TOPK/MMR - RANDOM same budget: {best_query_gap:.2f} "
        f"({best_query_row['Method']}_{best_query_row['Budget']})",
        "",
        "### TOPK - RANDOM",
        f"- k=2: {find_summary_row(rows, 'TOPK', 2)['Short F1'] - random_2['Short F1']:.2f}",
        f"- k=4: {find_summary_row(rows, 'TOPK', 4)['Short F1'] - random_4['Short F1']:.2f}",
        "",
        "### MMR - TOPK",
        *[f"- k={budget}: {gap:.2f}" for budget, gap in mmr_topk_gaps.items()],
        "",
        "## Decision criteria",
        "",
        f"- Packet representation pass (gap <= 8): {representation_pass}",
        f"- Projector first (gap > 15): {projector_needed}",
        f"- Query-aware selection signal (gain >= 5): {selection_signal}",
        f"- Controller space: {controller_space}",
        f"- STOP signal (best k in 2/3 and ALL lower by >= 5): {stop_signal}",
        f"- ORACLE_2 - ORACLE_1: {multihop_gain:.2f}",
        f"- NO_CONTEXT Short F1: {no_context['Short F1']:.2f}",
        f"- ORACLE_1 Short F1: {oracle_1['Short F1']:.2f}",
        f"- ORACLE_2 Short F1: {oracle_2['Short F1']:.2f}",
        f"- XRAG_ORACLE Short F1: {xrag_oracle['Short F1']:.2f}",
        f"- ALL Short F1: {all_packets['Short F1']:.2f}",
        "",
        "## Final recommendation",
        "",
        final_recommendation,
    ]

    comparisons = {
        "text_no_context": text_no_context,
        "representation_gap": representation_gap,
        "oracle_heuristic_gap": oracle_heuristic_gap,
        "best_heuristic": f"{best_heuristic['Method']}_{best_heuristic['Budget']}",
        "best_query_gap": best_query_gap,
        "representation_pass": representation_pass,
        "projector_needed": projector_needed,
        "selection_signal": selection_signal,
        "controller_space": controller_space,
        "stop_signal": stop_signal,
        "multihop_gain": multihop_gain,
        "final_recommendation": final_recommendation,
    }
    return "\n".join(lines) + "\n", comparisons


def print_required_comparisons(comparisons):
    print("", flush=True)
    print(f"TEXT_ORACLE - NO_CONTEXT: {comparisons['text_no_context']:.2f}", flush=True)
    print(f"TEXT_ORACLE - XRAG_ORACLE: {comparisons['representation_gap']:.2f}", flush=True)
    print(f"XRAG_ORACLE - best heuristic: {comparisons['oracle_heuristic_gap']:.2f}", flush=True)
    print(f"Best heuristic: {comparisons['best_heuristic']}", flush=True)
    print(f"Best TOPK/MMR - RANDOM: {comparisons['best_query_gap']:.2f}", flush=True)
    print(f"Representation pass: {comparisons['representation_pass']}", flush=True)
    print(f"Projector needed first: {comparisons['projector_needed']}", flush=True)
    print(f"Selection signal: {comparisons['selection_signal']}", flush=True)
    print(f"Controller space: {comparisons['controller_space']}", flush=True)
    print(f"STOP signal: {comparisons['stop_signal']}", flush=True)
    print(f"ORACLE_2 - ORACLE_1: {comparisons['multihop_gain']:.2f}", flush=True)
    print(f"Final recommendation: {comparisons['final_recommendation']}", flush=True)


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
                if method != "NO_CONTEXT":
                    assert selected_indices
                selected_supporting_count, support_recall, support_precision, full_support_coverage = coverage_metrics(
                    selected_indices,
                    gold_supporting_indices,
                )

                generate_start = time.perf_counter()
                if input_mode == "no_context":
                    prompt = build_no_context_prompt(question)
                    raw_prediction, input_tokens, generated_tokens = generate_text_answer(
                        xrag_tokenizer,
                        xrag_model,
                        prompt,
                        device,
                        args.max_new_tokens,
                    )
                elif input_mode == "text":
                    prompt = build_text_oracle_prompt(question, packets, selected_indices)
                    raw_prediction, input_tokens, generated_tokens = generate_text_answer(
                        xrag_tokenizer,
                        xrag_model,
                        prompt,
                        device,
                        args.max_new_tokens,
                    )
                else:
                    prompt = build_xrag_prompt(question, len(selected_indices))
                    selected_embeddings = packet_raw[selected_indices]
                    assert selected_embeddings.shape[0] == len(selected_indices)
                    raw_prediction, input_tokens, generated_tokens = generate_xrag_answer(
                        xrag_tokenizer,
                        xrag_model,
                        xrag_token_id,
                        question,
                        selected_embeddings,
                        device,
                        args.max_new_tokens,
                    )
                generate_seconds = time.perf_counter() - generate_start
                clean = clean_prediction(raw_prediction)
                short = extract_short_answer(raw_prediction)
                assert clean.strip()
                assert short.strip()

                clean_em, clean_f1 = score_prediction(clean, gold_answer)
                short_em, short_f1 = score_prediction(short, gold_answer)
                assert clean_em in {0.0, 1.0}
                assert short_em in {0.0, 1.0}
                assert 0.0 <= clean_f1 <= 1.0
                assert 0.0 <= short_f1 <= 1.0
                row = {
                    "sample_id": sample_id,
                    "method": method,
                    "budget": budget,
                    "random_seed": random_seed,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "raw_prediction": raw_prediction,
                    "clean_prediction": clean,
                    "short_prediction": short,
                    "clean_em": clean_em,
                    "clean_f1": clean_f1,
                    "short_em": short_em,
                    "short_f1": short_f1,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
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
                    f"short_em={short_em:.0f} short_f1={short_f1:.3f} "
                    f"clean_f1={clean_f1:.3f} gen={generate_seconds:.3f}s "
                    f"pred={short[:80]!r}",
                    flush=True,
                )

            required_once = {
                "NO_CONTEXT", "TEXT_ORACLE", "XRAG_ORACLE", "ORACLE_1",
                "ORACLE_2", "ALL", "TOPK_1", "TOPK_2", "TOPK_3",
                "TOPK_4", "MMR_1", "MMR_2", "MMR_3", "MMR_4",
            }
            assert required_once.issubset(set(per_sample_methods))
            assert per_sample_methods.count("RANDOM_2") == 3
            assert per_sample_methods.count("RANDOM_4") == 3

    elapsed = time.perf_counter() - start_all
    print(f"\nTotal runtime seconds: {elapsed:.3f}", flush=True)
    print(f"Peak allocated GB: {vram_gb(torch.cuda.max_memory_allocated):.3f}", flush=True)
    print(f"Peak reserved GB: {vram_gb(torch.cuda.max_memory_reserved):.3f}", flush=True)
    print(f"Predictions written to: {output_path}", flush=True)

    expected_rows = len(dataset) * 20
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
