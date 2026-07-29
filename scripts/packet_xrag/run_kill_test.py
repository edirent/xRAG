#!/usr/bin/env python
import argparse
import csv
import json
import re
import string
import sys
import time
from collections import defaultdict
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


def parse_args():
    parser = argparse.ArgumentParser(description="Packet-xRAG HotpotQA kill test.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default="cache/results/killtest_predictions.jsonl")
    parser.add_argument("--summary-output", default="cache/results/killtest_summary.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 12])
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
    assert any(p["is_supporting"] for p in packets)
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


def select_all(packets, budget=None):
    return list(range(len(packets)))


def select_oracle(packets, budget=None):
    return [i for i, packet in enumerate(packets) if packet["is_supporting"]]


def select_topk(packet_norm, query_norm, k):
    relevance = packet_norm @ query_norm
    selected = torch.topk(relevance, k=min(k, packet_norm.shape[0])).indices.tolist()
    selected.sort()
    return selected


def select_mmr(packet_norm, query_norm, k, mmr_lambda):
    selected = []
    remaining = set(range(packet_norm.shape[0]))

    while remaining and len(selected) < k:
        best_index = None
        best_score = float("-inf")

        for i in remaining:
            relevance_score = float(packet_norm[i] @ query_norm)
            if not selected:
                redundancy_score = 0.0
            else:
                redundancy_score = max(float(packet_norm[i] @ packet_norm[j]) for j in selected)

            score = mmr_lambda * relevance_score - (1.0 - mmr_lambda) * redundancy_score
            if score > best_score:
                best_score = score
                best_index = i

        selected.append(best_index)
        remaining.remove(best_index)

    return sorted(selected)


def build_prompt(question, num_packets):
    background_tokens = " ".join([XRAG_TOKEN] * num_packets)
    return (
        "Refer to the background information and answer "
        "the question with a short answer.\n\n"
        f"Background: {background_tokens}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


@torch.no_grad()
def generate_answer(
    tokenizer,
    model,
    xrag_token_id,
    question,
    selected_embeddings,
    device,
    max_new_tokens,
):
    prompt = build_prompt(question, selected_embeddings.shape[0])
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


def normalize_answer(text):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
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


def method_runs(packets, packet_norm, query_norm, budgets, mmr_lambda):
    yield "ALL", None, select_all(packets)
    yield "ORACLE", None, select_oracle(packets)
    for budget in budgets:
        yield "TOPK", budget, select_topk(packet_norm, query_norm, budget)
    for budget in budgets:
        yield "MMR", budget, select_mmr(packet_norm, query_norm, budget, mmr_lambda)


def write_summary(predictions, summary_output):
    groups = defaultdict(list)
    all_packet_counts = []
    oracle_packet_counts = []
    for row in predictions:
        groups[(row["method"], row["budget"])].append(row)
        if row["method"] == "ALL":
            all_packet_counts.append(row["total_packets"])
        elif row["method"] == "ORACLE":
            oracle_packet_counts.append(row["num_packets"])

    average_all_packets = sum(all_packet_counts) / len(all_packet_counts)
    average_oracle_packets = sum(oracle_packet_counts) / len(oracle_packet_counts)

    ordered_keys = [("ALL", None), ("ORACLE", None)]
    budgets = sorted({budget for method, budget in groups if budget is not None})
    ordered_keys.extend(("TOPK", budget) for budget in budgets)
    ordered_keys.extend(("MMR", budget) for budget in budgets)

    rows = []
    for method, budget in ordered_keys:
        items = groups[(method, budget)]
        avg_em = sum(item["em"] for item in items) / len(items)
        avg_f1 = sum(item["f1"] for item in items) / len(items)
        avg_packets = sum(item["num_packets"] for item in items) / len(items)
        packet_ratio = avg_packets / average_all_packets
        rows.append(
            {
                "Method": method,
                "Budget": "variable" if method == "ORACLE" else ("" if budget is None else budget),
                "EM": avg_em,
                "F1": avg_f1,
                "Avg packets": avg_packets,
                "Packet ratio": packet_ratio,
            }
        )

    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "Budget", "EM", "F1", "Avg packets", "Packet ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary", flush=True)
    print("| Method | Budget | EM | F1 | Avg packets | Packet ratio |", flush=True)
    print("| --- | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for row in rows:
        print(
            f"| {row['Method']} | {row['Budget']} | "
            f"{100 * row['EM']:.2f} | {100 * row['F1']:.2f} | "
            f"{row['Avg packets']:.2f} | {row['Packet ratio']:.3f} |",
            flush=True,
        )

    all_f1 = next(row["F1"] for row in rows if row["Method"] == "ALL")
    oracle_f1 = next(row["F1"] for row in rows if row["Method"] == "ORACLE")
    topk_rows = [row for row in rows if row["Method"] == "TOPK"]
    mmr_rows = [row for row in rows if row["Method"] == "MMR"]
    best_topk = max(topk_rows, key=lambda row: row["F1"])
    best_mmr = max(mmr_rows, key=lambda row: row["F1"])

    print("", flush=True)
    print(f"Average total packets per sample: {average_all_packets:.2f}", flush=True)
    print(f"Average oracle packets per sample: {average_oracle_packets:.2f}", flush=True)
    print(f"ALL F1: {100 * all_f1:.2f}", flush=True)
    print(f"ORACLE F1: {100 * oracle_f1:.2f}", flush=True)
    print(
        "Best TOPK result: "
        f"k={best_topk['Budget']} EM={100 * best_topk['EM']:.2f} F1={100 * best_topk['F1']:.2f}",
        flush=True,
    )
    print(
        "Best MMR result: "
        f"k={best_mmr['Budget']} EM={100 * best_mmr['EM']:.2f} F1={100 * best_mmr['F1']:.2f}",
        flush=True,
    )

    return rows


def print_gate_decision(summary_rows):
    def find_row(method, budget=None):
        for row in summary_rows:
            if row["Method"] == method and (budget is None or row["Budget"] == budget):
                return row
        raise KeyError((method, budget))

    all_row = find_row("ALL")
    oracle_row = find_row("ORACLE")
    mmr_rows = [row for row in summary_rows if row["Method"] == "MMR"]
    all_f1 = all_row["F1"]
    oracle_f1 = oracle_row["F1"]
    oracle_ratio = oracle_row["Packet ratio"]
    avg_oracle_packets = oracle_row["Avg packets"]
    target_f1 = oracle_f1 - 0.02

    mmr_reaching_target = [row for row in mmr_rows if row["F1"] >= target_f1]
    min_mmr_target = min(mmr_reaching_target, key=lambda row: int(row["Budget"])) if mmr_reaching_target else None

    gate_a_pass = oracle_f1 >= all_f1 - 0.02
    gate_a_fail = oracle_f1 < all_f1 - 0.05
    gate_b_strong = oracle_ratio <= 0.25
    gate_b_fail = oracle_ratio > 0.40
    gate_c_strong = min_mmr_target is None or int(min_mmr_target["Budget"]) >= 1.5 * avg_oracle_packets
    gate_c_weak = min_mmr_target is not None and int(min_mmr_target["Budget"]) <= 1.2 * avg_oracle_packets

    print("\nKill-test gates", flush=True)
    print(f"Gate A packet representation pass: {gate_a_pass}", flush=True)
    print(f"Gate A packet representation fail: {gate_a_fail}", flush=True)
    print(f"Gate B compression strong pass: {gate_b_strong}", flush=True)
    print(f"Gate B compression fail: {gate_b_fail}", flush=True)
    if min_mmr_target is None:
        print("Gate C MMR target: not reached by k=12", flush=True)
    else:
        print(f"Gate C MMR target reached at k={min_mmr_target['Budget']}", flush=True)
    print(f"Gate C controller space strong: {gate_c_strong}", flush=True)
    print(f"Gate C direction weak: {gate_c_weak}", flush=True)

    if gate_a_pass and gate_b_strong and gate_c_strong:
        decision = "CONTINUE: construct controlled redundancy dataset and train marginal-value controller."
    elif all_f1 > 0 and gate_a_fail:
        decision = "PAUSE: fix packet representation / train packet-level projector before controller."
    elif gate_b_fail or gate_c_weak:
        decision = "KILL CURRENT ANGLE: insufficient compression gap or MMR nearly matches oracle budget."
    else:
        decision = "INCONCLUSIVE: inspect predictions before choosing controller or projector work."
    print(f"Decision: {decision}", flush=True)


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
    start_all = time.perf_counter()
    with output_path.open("w") as out:
        for sample_index, sample in enumerate(dataset):
            sample_id = str(sample.get("id", sample.get("_id", sample_index)))
            question = sample["question"]
            gold_answer = sample["answer"]
            packets = make_packets(sample)

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
                f"packets={len(packets)} oracle={sum(p['is_supporting'] for p in packets)} "
                f"encode={encode_seconds:.3f}s",
                flush=True,
            )

            for method, budget, selected_indices in method_runs(
                packets,
                packet_norm,
                query_norm,
                args.budgets,
                args.mmr_lambda,
            ):
                assert selected_indices
                selected_embeddings = packet_raw[selected_indices]
                assert selected_embeddings.shape[0] == len(selected_indices)

                generate_start = time.perf_counter()
                prediction = generate_answer(
                    xrag_tokenizer,
                    xrag_model,
                    xrag_token_id,
                    question,
                    selected_embeddings,
                    device,
                    args.max_new_tokens,
                )
                generate_seconds = time.perf_counter() - generate_start
                em, f1 = score_prediction(prediction, gold_answer)

                row = {
                    "sample_id": sample_id,
                    "method": method,
                    "budget": budget,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "num_packets": len(selected_indices),
                    "total_packets": len(packets),
                    "selected_indices": selected_indices,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                predictions.append(row)

                print(
                    f"  {method}-{budget if budget is not None else 'var'} "
                    f"n={len(selected_indices)} em={em:.0f} f1={f1:.3f} "
                    f"gen={generate_seconds:.3f}s pred={prediction[:80]!r}",
                    flush=True,
                )

    elapsed = time.perf_counter() - start_all
    print(f"\nTotal runtime seconds: {elapsed:.3f}", flush=True)
    print(f"Peak allocated GB: {vram_gb(torch.cuda.max_memory_allocated):.3f}", flush=True)
    print(f"Peak reserved GB: {vram_gb(torch.cuda.max_memory_reserved):.3f}", flush=True)
    print(f"Predictions written to: {output_path}", flush=True)

    summary_rows = write_summary(predictions, args.summary_output)
    print(f"Summary written to: {args.summary_output}", flush=True)
    print_gate_decision(summary_rows)


if __name__ == "__main__":
    main()
