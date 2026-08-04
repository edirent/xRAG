"""Frozen generation and metrics shared by all adapted QA datasets."""

from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean

import torch

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from src.packet_xrag.composition.fused_xrag_injection import greedy_generate_fused


def answer_scores(prediction, record):
    scores = [selector.score_prediction(prediction, answer)
              for answer in record.get("answers", [record["answer"]])]
    return max(value[0] for value in scores), max(value[1] for value in scores)


def decode_row(record, selected, generated, tokenizer, configuration, input_tokens,
               output_tokens, context_tokens, fuser_ms, total_ms, peak):
    eos = generated.eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
    length = int(eos[0]) + 1 if len(eos) else len(generated)
    raw = tokenizer.decode(generated[:length], skip_special_tokens=False)
    short = selector.extract_short_answer(raw) or "[EMPTY]"; clean = selector.clean_prediction(raw)
    em, f1 = answer_scores(short, record); _, clean_f1 = answer_scores(clean, record)
    gold, chosen = set(record["gold_packet_ids"]), set(selected)
    return {"sample_id": record["sample_id"], "configuration": configuration,
        "selected_packet_ids": list(selected), "short_prediction": short,
        "short_em": em, "short_f1": f1, "clean_f1": clean_f1,
        "is_empty": short == "[EMPTY]", "input_packets": len(selected),
        "input_packet_soft_tokens": input_tokens, "output_fused_tokens": output_tokens,
        "llm_context_evidence_tokens": context_tokens,
        "support_recall": len(gold & chosen) / len(gold) if gold else 0.0,
        "full_support": float(gold.issubset(chosen)) if gold else 0.0,
        "has_support_labels": bool(gold), "fuser_latency_ms": fuser_ms,
        "total_latency_ms": total_ms, "peak_vram_gb": peak,
        "metadata": record.get("metadata", {})}


def summarize(rows):
    labeled = [row for row in rows if row["has_support_labels"]]
    return {"samples": len(rows), "short_f1": 100 * mean(row["short_f1"] for row in rows),
        "short_em": 100 * mean(row["short_em"] for row in rows),
        "clean_f1": 100 * mean(row["clean_f1"] for row in rows),
        "empty": sum(row["is_empty"] for row in rows),
        "input_packets": mean(row["input_packets"] for row in rows),
        "input_packet_soft_tokens": mean(row["input_packet_soft_tokens"] for row in rows),
        "output_fused_tokens": mean(row["output_fused_tokens"] for row in rows),
        "llm_context_evidence_tokens": mean(row["llm_context_evidence_tokens"] for row in rows),
        "support_labeled_samples": len(labeled),
        "support_recall": mean(row["support_recall"] for row in labeled) if labeled else None,
        "full_support": mean(row["full_support"] for row in labeled) if labeled else None,
        "mean_fuser_latency_ms": mean(row["fuser_latency_ms"] for row in rows),
        "mean_total_latency_ms": mean(row["total_latency_ms"] for row in rows),
        "peak_vram_gb": max(row["peak_vram_gb"] for row in rows)}


@torch.inference_mode()
def evaluate_independent(configuration, records, groups, k2, tokenizer, generator,
                         xrag_id, device, batch_size=8):
    rows, buckets = [], defaultdict(list)
    for record, selected in zip(records, groups): buckets[len(selected)].append((record, selected))
    for count, items in sorted(buckets.items()):
        if count == 0: raise ValueError("use evaluate_no_context for empty evidence")
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]; torch.cuda.reset_peak_memory_stats(device)
            began = time.time()
            projected = torch.stack([k2(record["packet_embeddings"][selected].to(
                device=device, dtype=torch.bfloat16)).reshape(2 * count, 4096)
                for record, selected in batch])
            prompts = tokenizer([v1.build_prompt(record["question"], 2 * count)
                                 for record, _ in batch], return_tensors="pt",
                                add_special_tokens=False, padding=True).to(device)
            generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
                prompts.attention_mask, xrag_id, projected, max_new_tokens=32)
            torch.cuda.synchronize(device); elapsed = 1000 * (time.time() - began) / len(batch)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            rows.extend(decode_row(record, selected, generated[index], tokenizer, configuration,
                2 * count, 2 * count, 2 * count, 0.0, elapsed, peak)
                for index, (record, selected) in enumerate(batch))
    return rows


@torch.inference_mode()
def evaluate_fuser(configuration, fuser, records, groups, make_fused, k2, tokenizer,
                   generator, xrag_id, device, batch_size=8):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; selected = groups[start:start + batch_size]
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); began = time.time()
        fused = make_fused(fuser, batch, selected, k2, device)
        torch.cuda.synchronize(device); fuser_ms = 1000 * (time.time() - began) / len(batch)
        prompts = tokenizer([v1.build_prompt(record["question"], 4) for record in batch],
                            return_tensors="pt", add_special_tokens=False, padding=True).to(device)
        generated = greedy_generate_fused(generator, tokenizer, prompts.input_ids,
            prompts.attention_mask, xrag_id, fused, max_new_tokens=32)
        torch.cuda.synchronize(device); total_ms = 1000 * (time.time() - began) / len(batch)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.extend(decode_row(record, group, generated[index], tokenizer, configuration,
            2 * len(group), 4, 4, fuser_ms, total_ms, peak)
            for index, (record, group) in enumerate(zip(batch, selected)))
    return rows


def text_prompt(question, packets):
    background = "\n".join(packet["encoder_text"] for packet in packets)
    content = ("Refer to the background document and answer the question. Respond only with the "
               "shortest possible answer. Do not provide an explanation.\n\nBackground: "
               f"{background}\n\nQuestion: {question}")
    return f"[INST] {content} [/INST] The answer is:"


@torch.inference_mode()
def evaluate_text(configuration, records, groups, tokenizer, generator, device, batch_size=4):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; selected = groups[start:start + batch_size]
        prompts = [text_prompt(record["question"], [record["packets"][i] for i in group])
                   for record, group in zip(batch, selected)]
        tokenized = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                              padding=True).to(device); began = time.time()
        output = generator.generate(input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask, do_sample=False, max_new_tokens=32,
            use_cache=True, pad_token_id=tokenizer.pad_token_id)
        generated = output[:, tokenized.input_ids.shape[1]:]
        torch.cuda.synchronize(device); elapsed = 1000 * (time.time() - began) / len(batch)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.extend(decode_row(record, group, generated[index], tokenizer, configuration,
            0, 0, int(tokenized.attention_mask[index].sum()), 0.0, elapsed, peak)
            for index, (record, group) in enumerate(zip(batch, selected)))
    return rows


@torch.inference_mode()
def evaluate_no_context(records, tokenizer, generator, device, batch_size=8):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prompts = tokenizer([v1.build_prompt(record["question"], 0) for record in batch],
                            return_tensors="pt", add_special_tokens=False, padding=True).to(device)
        began = time.time(); output = generator.generate(input_ids=prompts.input_ids,
            attention_mask=prompts.attention_mask, do_sample=False, max_new_tokens=32,
            use_cache=True, pad_token_id=tokenizer.pad_token_id)
        generated = output[:, prompts.input_ids.shape[1]:]
        torch.cuda.synchronize(device); elapsed = 1000 * (time.time() - began) / len(batch)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.extend(decode_row(record, [], generated[index], tokenizer, "NO_CONTEXT",
            0, 0, 0, 0.0, elapsed, peak) for index, record in enumerate(batch))
    return rows

