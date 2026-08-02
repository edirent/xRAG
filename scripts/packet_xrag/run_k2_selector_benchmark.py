#!/usr/bin/env python
"""Frozen-K2 selector benchmark on the locked 500-example validation split."""

import argparse
import csv
import hashlib
import inspect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.token_resampler_common import (
    EXPECTED_K2_SHA256,
    EXPECTED_SPLIT_HASH,
    EXPECTED_V1_SHA256,
    POOLED_K2_F1,
    audit_checkpoints,
    load_frozen_k2_model,
    locked_records,
    sha256_file,
)
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector


RANDOM_SEEDS = (13, 37, 73)
SUMMARY_FIELDS = [
    "Configuration", "Budget", "Random Seed", "Short EM", "Short F1",
    "Clean F1", "Substring", "Support Recall", "Full Support", "Avg Packets",
    "Avg Soft Tokens", "EMPTY",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--validation-ids-file", default="cache/results/packet_representation_ablation_validation_ids.json")
    parser.add_argument("--split-file", default="cache/projector/packet_projector_calibration/data_split.json")
    parser.add_argument("--v1-training-config", default="cache/projector/packet_projector_calibration/last/training_config.json")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--v1-projector-checkpoint", default=None)
    parser.add_argument("--k2-projector-checkpoint", default=None)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(RANDOM_SEEDS))
    parser.add_argument("--max-budget", type=int, default=6)
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="cache/results/k2_selector_benchmark_full500.jsonl")
    parser.add_argument("--summary-output", default="cache/results/k2_selector_benchmark_full500_summary.csv")
    parser.add_argument("--decision-output", default="cache/results/k2_selector_benchmark_decision.md")
    parser.add_argument("--audit-json", default="cache/results/k2_selector_checkpoint_audit.json")
    parser.add_argument("--audit-md", default="cache/results/k2_selector_checkpoint_audit.md")
    return parser.parse_args(argv)


def stable_random_indices(num_packets, budget, sample_id, seed):
    digest = hashlib.sha256(f"{sample_id}:{seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return sorted(rng.sample(range(num_packets), min(budget, num_packets)))


def topk_selection(relevance, budget):
    return sorted(range(len(relevance)), key=lambda index: (-float(relevance[index]), index))[:min(budget, len(relevance))]


def mmr_selection(relevance, packet_norm, budget, mmr_lambda=0.5):
    if mmr_lambda != 0.5:
        raise ValueError("the locked protocol requires MMR lambda=0.5")
    selected, details = [], {}
    remaining = set(range(len(relevance)))
    while remaining and len(selected) < budget:
        candidates = []
        for index in remaining:
            maximum_similarity = (max(float(packet_norm[index] @ packet_norm[chosen]) for chosen in selected)
                                  if selected else 0.0)
            score = (float(relevance[index]) if not selected else
                     mmr_lambda * float(relevance[index]) - (1.0 - mmr_lambda) * maximum_similarity)
            candidates.append((-score, -float(relevance[index]), index, score, maximum_similarity))
        _, _, chosen, score, maximum_similarity = min(candidates)
        details[chosen] = {
            "query_relevance": float(relevance[chosen]),
            "mmr_score_at_selection": score,
            "max_similarity_to_selected": maximum_similarity,
            "selection_step": len(selected) + 1,
        }
        selected.append(chosen); remaining.remove(chosen)
    return selected, details


def make_candidate_packets(sample):
    supporting_pairs = list(zip(sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"]))
    packets = []
    for doc_id, (title, sentences) in enumerate(zip(sample["context"]["title"], sample["context"]["sentences"])):
        for sentence_id, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            packets.append({
                "packet_id": len(packets), "doc_id": doc_id, "title": title,
                "sentence_id": sentence_id, "text": sentence,
                "encoder_text": f"[{title}] {sentence}",
                "is_supporting": (title, sentence_id) in set(supporting_pairs),
            })
    pair_to_ids = defaultdict(list)
    for packet in packets:
        pair_to_ids[(packet["title"], packet["sentence_id"])].append(packet["packet_id"])
    missing, ambiguous, gold_ids = [], [], []
    for pair in supporting_pairs:
        matches = pair_to_ids.get(pair, [])
        if not matches: missing.append(pair)
        elif len(matches) != 1: ambiguous.append((pair, matches))
        elif matches[0] not in gold_ids: gold_ids.append(matches[0])
    if missing or ambiguous:
        raise ValueError(f"gold mapping failed; missing={missing}, ambiguous={ambiguous}")
    if not packets or not gold_ids:
        raise ValueError("empty candidate or gold packet set")
    return packets, gold_ids


def oracle_selection(gold_ids, budget=None):
    return list(gold_ids if budget is None else gold_ids[:budget])


def method_configurations(sample_id, packets, gold_ids, relevance, packet_norm, seeds, max_budget, mmr_lambda):
    yield {"configuration": "NO_CONTEXT", "method_family": "NO_CONTEXT", "budget": None,
           "random_seed": None, "selected": [], "details": {}, "mode": "no_context"}
    yield {"configuration": "TEXT_ORACLE", "method_family": "TEXT_ORACLE", "budget": None,
           "random_seed": None, "selected": oracle_selection(gold_ids), "details": {}, "mode": "text"}
    yield {"configuration": "ORACLE_1", "method_family": "ORACLE", "budget": 1,
           "random_seed": None, "selected": oracle_selection(gold_ids, 1), "details": {}, "mode": "xrag"}
    yield {"configuration": "ORACLE_2", "method_family": "ORACLE", "budget": 2,
           "random_seed": None, "selected": oracle_selection(gold_ids, 2), "details": {}, "mode": "xrag"}
    yield {"configuration": "ALL", "method_family": "ALL", "budget": None,
           "random_seed": None, "selected": list(range(len(packets))), "details": {}, "mode": "xrag"}
    for budget in range(1, max_budget + 1):
        for seed in seeds:
            yield {"configuration": f"RANDOM_{budget}_SEED{seed}", "method_family": "RANDOM",
                   "budget": budget, "random_seed": seed,
                   "selected": stable_random_indices(len(packets), budget, sample_id, seed),
                   "details": {}, "mode": "xrag"}
    for budget in range(1, max_budget + 1):
        selected = topk_selection(relevance, budget)
        details = {index: {"query_relevance": float(relevance[index])} for index in selected}
        yield {"configuration": f"TOPK_{budget}", "method_family": "TOPK", "budget": budget,
               "random_seed": None, "selected": selected, "details": details, "mode": "xrag"}
    for budget in range(1, max_budget + 1):
        selected, details = mmr_selection(relevance, packet_norm, budget, mmr_lambda)
        yield {"configuration": f"MMR_{budget}", "method_family": "MMR", "budget": budget,
               "random_seed": None, "selected": selected, "details": details, "mode": "xrag"}


def build_text_prompt(question, packets, selected):
    background = "\n".join(packets[index]["encoder_text"] for index in selected)
    content = ("Refer to the background document and answer the question. "
               "Respond only with the shortest possible answer. "
               "Do not provide an explanation.\n\n"
               f"Background: {background}\n\nQuestion: {question}")
    return f"[INST] {content} [/INST] The answer is:"


def build_no_context_prompt(question):
    content = ("Refer to the background document and answer the question. "
               "Respond only with the shortest possible answer. "
               "Do not provide an explanation.\n\nBackground: \n\n"
               f"Question: {question}")
    return f"[INST] {content} [/INST] The answer is:"


@torch.inference_mode()
def encode_for_ranking(tokenizer, sfr, question, packets, device):
    # This is the exact historical selector-calibration query protocol: the raw
    # question is encoded as a document, with no query instruction.
    texts = [question] + [packet["encoder_text"] for packet in packets]
    tokenized = tokenizer(texts, max_length=180, padding=True, truncation=True, return_tensors="pt").to(device)
    embeddings = sfr.get_doc_embedding(tokenized.input_ids, tokenized.attention_mask).view(len(texts), -1)
    query_norm = F.normalize(embeddings[0].float(), dim=-1)
    packet_norm = F.normalize(embeddings[1:].float(), dim=-1)
    relevance = packet_norm @ query_norm
    return relevance.cpu(), packet_norm.cpu()


@torch.inference_mode()
def encode_selected(tokenizer, sfr, packets, selected, device):
    return v1.encode_packets(tokenizer, sfr, [packets[index]["encoder_text"] for index in selected], device)


@torch.inference_mode()
def generate_text(tokenizer, model, prompt, device, max_new_tokens):
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generated = model.generate(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask,
                               do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                               pad_token_id=tokenizer.pad_token_id)
    new = generated[:, tokenized.input_ids.shape[1]:]
    return tokenizer.batch_decode(new, skip_special_tokens=False)[0], tokenized.input_ids.shape[1], new.shape[1]


@torch.inference_mode()
def generate_xrag(tokenizer, model, xrag_id, question, embeddings, device, max_new_tokens):
    prompt = v1.build_prompt(question, embeddings.shape[0] * 2)
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    num_packets = embeddings.shape[0]
    assert model.projector(embeddings.to(next(model.parameters()).dtype)).shape == (num_packets, 2, model.config.hidden_size)
    assert int((tokenized.input_ids == xrag_id).sum()) == num_packets * 2
    generated = model.generate(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask,
                               retrieval_embeds=embeddings, do_sample=False, max_new_tokens=max_new_tokens,
                               use_cache=True, pad_token_id=tokenizer.pad_token_id)
    new = generated[:, tokenized.input_ids.shape[1]:] if generated.shape[1] > tokenized.input_ids.shape[1] else generated
    return tokenizer.batch_decode(new, skip_special_tokens=False)[0], tokenized.input_ids.shape[1], new.shape[1]


def substring_score(prediction, gold):
    pred, target = selector.normalize_answer(prediction), selector.normalize_answer(gold)
    return float(bool(pred) and (pred in target or target in pred))


def make_result(sample, packets, gold_ids, config, raw, prompt_tokens, generated_tokens):
    selected = config["selected"]; gold_set = set(gold_ids); selected_set = set(selected)
    support_recall = len(gold_set & selected_set) / len(gold_set)
    full_support = float(gold_set.issubset(selected_set))
    clean = selector.clean_prediction(raw); short = selector.extract_short_answer(raw) or "[EMPTY]"
    em, f1 = selector.score_prediction(short, sample["answer"]); _, clean_f1 = selector.score_prediction(clean, sample["answer"])
    selected_packets = []
    for index in selected:
        packet = dict(packets[index]); packet.update(config["details"].get(index, {})); selected_packets.append(packet)
    xrag_mode = config["mode"] == "xrag"
    return {
        "sample_id": str(sample["id"]), "configuration": config["configuration"],
        "method_family": config["method_family"], "budget": config["budget"],
        "random_seed": config["random_seed"], "question": sample["question"],
        "gold_answer": sample["answer"], "num_candidate_packets": len(packets),
        "num_gold_packets": len(gold_ids), "gold_packet_ids": gold_ids,
        "selected_packet_ids": selected, "selected_packets": selected_packets,
        "support_recall": support_recall, "full_support_coverage": full_support,
        "num_packets": len(selected), "tokens_per_packet": 2 if xrag_mode else 0,
        "total_soft_tokens": len(selected) * 2 if xrag_mode else 0,
        "prompt_tokens": prompt_tokens, "raw_prediction": raw,
        "clean_prediction": clean, "short_prediction": short, "short_em": em,
        "short_f1": f1, "clean_f1": clean_f1,
        "substring_match": substring_score(short, sample["answer"]),
        "generated_tokens": generated_tokens, "is_empty": short == "[EMPTY]",
    }


def summarize(rows, seeds):
    groups = defaultdict(list)
    for row in rows: groups[row["configuration"]].append(row)
    summaries = []
    for configuration, items in groups.items():
        first = items[0]
        summaries.append({
            "Configuration": configuration, "Budget": "" if first["budget"] is None else first["budget"],
            "Random Seed": "" if first["random_seed"] is None else first["random_seed"],
            "Short EM": 100 * mean(item["short_em"] for item in items),
            "Short F1": 100 * mean(item["short_f1"] for item in items),
            "Clean F1": 100 * mean(item["clean_f1"] for item in items),
            "Substring": 100 * mean(item["substring_match"] for item in items),
            "Support Recall": mean(item["support_recall"] for item in items),
            "Full Support": mean(item["full_support_coverage"] for item in items),
            "Avg Packets": mean(item["num_packets"] for item in items),
            "Avg Soft Tokens": mean(item["total_soft_tokens"] for item in items),
            "EMPTY": sum(item["is_empty"] for item in items),
        })
    for budget in range(1, 7):
        seed_rows = [next(row for row in summaries if row["Configuration"] == f"RANDOM_{budget}_SEED{seed}") for seed in seeds]
        aggregate = {field: mean(row[field] for row in seed_rows) for field in
                     ["Short EM", "Short F1", "Clean F1", "Substring", "Support Recall", "Full Support", "Avg Packets", "Avg Soft Tokens", "EMPTY"]}
        summaries.append({"Configuration": f"RANDOM_{budget}_MEAN", "Budget": budget, "Random Seed": "MEAN", **aggregate})
        summaries.append({"Configuration": f"RANDOM_{budget}_STD", "Budget": budget, "Random Seed": "STD",
                          **{field: pstdev(row[field] for row in seed_rows) for field in aggregate}})
    order = {name: index for index, name in enumerate(
        ["NO_CONTEXT", "TEXT_ORACLE", "XRAG_ORACLE", "ORACLE_1", "ORACLE_2", "ALL"] +
        [f"RANDOM_{k}_SEED{s}" for k in range(1, 7) for s in seeds] +
        [f"RANDOM_{k}_MEAN" for k in range(1, 7)] + [f"RANDOM_{k}_STD" for k in range(1, 7)] +
        [f"TOPK_{k}" for k in range(1, 7)] + [f"MMR_{k}" for k in range(1, 7)])}
    return sorted(summaries, key=lambda row: order[row["Configuration"]])


def select_best_heuristic(summaries):
    candidates = [row for row in summaries if row["Configuration"].startswith(("TOPK_", "MMR_"))]
    best_f1 = max(row["Short F1"] for row in candidates)
    close = [row for row in candidates if best_f1 - row["Short F1"] < 0.25]
    close.sort(key=lambda row: (row["Avg Packets"], 0 if row["Configuration"].startswith("TOPK_") else 1, int(row["Budget"])))
    return close[0]


def write_summaries(rows, summary_path, seeds):
    summaries = summarize(rows, seeds); path = Path(summary_path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS); writer.writeheader(); writer.writerows(summaries)
    md_path = path.with_suffix(".md")
    lines = ["# Frozen-K2 Selector Benchmark Summary", "",
             "| Configuration | Budget | Short EM | Short F1 | Clean F1 | Substring | Support Recall | Full Support | Avg Packets | Avg Soft Tokens | EMPTY |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['Configuration']} | {row['Budget']} | {row['Short EM']:.4f} | {row['Short F1']:.4f} | {row['Clean F1']:.4f} | {row['Substring']:.4f} | {row['Support Recall']:.4f} | {row['Full Support']:.4f} | {row['Avg Packets']:.4f} | {row['Avg Soft Tokens']:.4f} | {row['EMPTY']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n")
    curve_path = path.parent / "k2_selector_budget_curve.csv"
    curve_fields = ["Method", "Budget", "Short F1", "Support Recall", "Full Support", "Avg Packets", "Avg Soft Tokens"]
    curves = []
    for method in ("RANDOM", "TOPK", "MMR"):
        for budget in range(1, 7):
            name = f"RANDOM_{budget}_MEAN" if method == "RANDOM" else f"{method}_{budget}"
            row = next(item for item in summaries if item["Configuration"] == name)
            curves.append({"Method": method, "Budget": budget, **{field: row[field] for field in curve_fields[2:]}})
    with curve_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_fields); writer.writeheader(); writer.writerows(curves)
    return summaries


def write_audit(args, hashes, tokenizer, xrag_id, model, sfr, v1_path, k2_path):
    k2_state = torch.load(k2_path, map_location="cpu", weights_only=True)
    v1_state = torch.load(v1_path, map_location="cpu", weights_only=True)
    embedded_equal = all(torch.equal(k2_state[f"base_projector.{name}"], tensor) for name, tensor in v1_state.items())
    if not embedded_equal: raise RuntimeError("K2 embedded V1 tensors differ from formal V1")
    forbidden = [name for name, _ in model.named_modules() if any(term in name.lower() for term in ("lora", "residual", "token_state"))]
    if forbidden or not isinstance(model.projector, MultiTokenPacketProjector):
        raise RuntimeError(f"unexpected adapters loaded: {forbidden}")
    audit = {
        "v1_checkpoint_path": str(Path(v1_path).resolve()), "v1_sha256": hashes["v1"],
        "k2_checkpoint_path": str(Path(k2_path).resolve()), "k2_sha256": hashes["k2"],
        "k2_embedded_v1_exact": embedded_equal, "sfr_model_identifier": v1.SFR_MODEL_NAME,
        "llm_model_identifier": v1.XRAG_MODEL_NAME, "tokenizer_identifier": v1.XRAG_MODEL_NAME,
        "xrag_token_id": xrag_id, "tokens_per_packet": 2,
        "validation_ids_file": str(Path(args.validation_ids_file).resolve()),
        "validation_split_hash": EXPECTED_SPLIT_HASH,
        "prompt_hash": hashlib.sha256(inspect.getsource(v1.build_prompt).encode()).hexdigest(),
        "answer_extractor_source_hash": hashlib.sha256(inspect.getsource(selector.extract_short_answer).encode()).hexdigest(),
        "query_text_template": "{question} (verbatim; no instruction)",
        "generation": {"do_sample": False, "max_new_tokens": 32, "use_cache": True, "decoding": "greedy"},
        "base_model_compatible": True, "unexpected_adapters_loaded": forbidden,
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sfr_trainable_parameters": sum(p.numel() for p in sfr.parameters() if p.requires_grad),
        "optimizer_created": False,
    }
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    Path(args.audit_md).write_text("# K2 Selector Checkpoint Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n")
    return audit


def validate_ids(args, validation_records):
    record = json.loads(Path(args.validation_ids_file).read_text()); ids = record["sample_ids"]
    if len(ids) != 500 or len(set(ids)) != 500 or record["split_hash"] != EXPECTED_SPLIT_HASH:
        raise RuntimeError("invalid locked validation ID file")
    actual = [str(sample["id"]) for sample, _, _ in validation_records]
    if actual != ids: raise RuntimeError("locked validation sample order mismatch")
    return ids


def print_debug(rows, sample_ids):
    by_sample = defaultdict(list)
    for row in rows: by_sample[row["sample_id"]].append(row)
    for sample_id in sample_ids[:3]:
        items = by_sample[sample_id]; first = items[0]
        print(json.dumps({
            "sample_id": sample_id, "question": first["question"], "gold_answer": first["gold_answer"],
            "gold_packet_ids": first["gold_packet_ids"],
            "topk": {row["configuration"]: row["selected_packet_ids"] for row in items if row["method_family"] == "TOPK"},
            "mmr": {row["configuration"]: row["selected_packet_ids"] for row in items if row["method_family"] == "MMR"},
            "random": {row["configuration"]: row["selected_packet_ids"] for row in items if row["method_family"] == "RANDOM"},
            "xrag_oracle": next(row["short_prediction"] for row in items if row["configuration"] == "XRAG_ORACLE"),
            "all": next(row["short_prediction"] for row in items if row["configuration"] == "ALL"),
            "short_f1": {row["configuration"]: row["short_f1"] for row in items},
        }, ensure_ascii=False, indent=2), flush=True)


@torch.inference_mode()
def main():
    args = parse_args()
    if tuple(args.random_seeds) != RANDOM_SEEDS or args.max_budget != 6 or args.mmr_lambda != 0.5:
        raise RuntimeError("selector protocol differs from locked configuration")
    if args.max_samples not in (10, 500): raise RuntimeError("only locked debug-10 or full-500 runs are allowed")
    k2_config = json.loads(Path(args.k2_training_config).read_text())
    v1_path = args.v1_projector_checkpoint or k2_config["base_projector_checkpoint"]
    k2_path = args.k2_projector_checkpoint or str(Path(k2_config["output_dir"]) / "best_short_f1" / "multi_token_projector.pt")
    hashes, _ = audit_checkpoints(v1_path, k2_path)
    if hashes != {"v1": EXPECTED_V1_SHA256, "k2": EXPECTED_K2_SHA256}: raise RuntimeError("formal checkpoint hash mismatch")
    train_records, validation_records = locked_records(args.split_file, args.v1_training_config)
    del train_records
    validate_ids(args, validation_records); records = validation_records[:args.max_samples]
    mapping_errors = []
    prepared = []
    for sample, _, _ in records:
        try: prepared.append((sample, *make_candidate_packets(sample)))
        except ValueError as error: mapping_errors.append({"sample_id": str(sample["id"]), "reason": str(error)})
    if mapping_errors:
        Path("cache/results/k2_selector_gold_mapping_errors.json").write_text(json.dumps(mapping_errors, indent=2) + "\n")
        raise RuntimeError(f"gold mapping failed for {len(mapping_errors)} samples")

    device = torch.device(args.device); torch.cuda.set_device(device)
    sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(v1.XRAG_MODEL_NAME, padding_side="left", add_eos_token=False, use_fast=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    xrag_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model, _ = load_frozen_k2_model(device, tokenizer, xrag_id, v1_path, k2_path)
    for parameter in model.parameters(): parameter.requires_grad = False
    for parameter in sfr.parameters(): parameter.requires_grad = False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in sfr.parameters())
    audit = write_audit(args, hashes, tokenizer, xrag_id, model, sfr, v1_path, k2_path)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); rows = []
    with output.open("w") as stream:
        # Mandatory full-500 XRAG oracle reproduction is completed first.
        for index, (sample, packets, gold_ids) in enumerate(prepared):
            embeddings = encode_selected(sfr_tokenizer, sfr, packets, gold_ids, device)
            raw, prompt_tokens, generated_tokens = generate_xrag(tokenizer, model, xrag_id, sample["question"], embeddings, device, args.max_new_tokens)
            config = {"configuration": "XRAG_ORACLE", "method_family": "ORACLE", "budget": None,
                      "random_seed": None, "selected": gold_ids, "details": {}, "mode": "xrag"}
            row = make_result(sample, packets, gold_ids, config, raw, prompt_tokens, generated_tokens)
            rows.append(row); stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            if (index + 1) % 50 == 0: print(f"XRAG_ORACLE {index+1}/{len(prepared)}", flush=True)
        observed_oracle = 100 * mean(row["short_f1"] for row in rows)
        reproduced = abs(observed_oracle - POOLED_K2_F1) <= 0.1
        audit["expected_xrag_oracle_short_f1"] = POOLED_K2_F1
        audit["observed_xrag_oracle_short_f1"] = observed_oracle
        audit["xrag_oracle_reproduced"] = reproduced
        Path(args.audit_json).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        if args.max_samples == 500 and not reproduced:
            raise RuntimeError(f"XRAG_ORACLE reproduction failed: {observed_oracle}")

        all_context_audit = []
        for sample_index, (sample, packets, gold_ids) in enumerate(prepared):
            relevance, packet_norm = encode_for_ranking(sfr_tokenizer, sfr, sample["question"], packets, device)
            embedding_cache = {}
            configs = method_configurations(str(sample["id"]), packets, gold_ids, relevance, packet_norm,
                                            tuple(args.random_seeds), args.max_budget, args.mmr_lambda)
            for config in configs:
                selected, mode = config["selected"], config["mode"]
                if mode == "no_context":
                    raw, prompt_tokens, generated_tokens = generate_text(tokenizer, model, build_no_context_prompt(sample["question"]), device, args.max_new_tokens)
                elif mode == "text":
                    raw, prompt_tokens, generated_tokens = generate_text(tokenizer, model, build_text_prompt(sample["question"], packets, selected), device, args.max_new_tokens)
                else:
                    cache_key = tuple(selected)
                    if cache_key not in embedding_cache:
                        embedding_cache[cache_key] = encode_selected(sfr_tokenizer, sfr, packets, selected, device)
                    embeddings = embedding_cache[cache_key]
                    raw, prompt_tokens, generated_tokens = generate_xrag(tokenizer, model, xrag_id, sample["question"], embeddings, device, args.max_new_tokens)
                    if config["configuration"] == "ALL":
                        total = prompt_tokens + args.max_new_tokens
                        all_context_audit.append({"sample_id": str(sample["id"]), "prompt_tokens": prompt_tokens,
                                                  "soft_tokens": len(selected) * 2, "total_with_generation": total,
                                                  "model_limit": model.config.max_position_embeddings,
                                                  "would_truncate": total > model.config.max_position_embeddings})
                row = make_result(sample, packets, gold_ids, config, raw, prompt_tokens, generated_tokens)
                rows.append(row); stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            print(f"sample {sample_index+1}/{len(prepared)} rows={len(rows)}", flush=True)
    if any(row["would_truncate"] for row in all_context_audit):
        Path("cache/results/k2_selector_all_context_audit.json").write_text(json.dumps(all_context_audit, indent=2) + "\n")
        raise RuntimeError("ALL contains context-limit violations")
    Path("cache/results/k2_selector_all_context_audit.json").write_text(json.dumps(all_context_audit, indent=2) + "\n")
    expected_rows = args.max_samples * (1 + 5 + 18 + 6 + 6)
    if len(rows) != expected_rows or len({row["sample_id"] for row in rows}) != args.max_samples:
        raise RuntimeError(f"benchmark cardinality mismatch: {len(rows)} != {expected_rows}")
    summaries = write_summaries(rows, args.summary_output, tuple(args.random_seeds))
    best = select_best_heuristic(summaries)
    preliminary = ["# Frozen-K2 Selector Benchmark Decision", "",
                   f"- Validation split hash: `{EXPECTED_SPLIT_HASH}`",
                   f"- XRAG_ORACLE observed: {observed_oracle:.6f}; reproduced: {reproduced}",
                   f"- Preliminary best heuristic: {best['Configuration']} at budget {best['Budget']}, F1 {best['Short F1']:.6f}",
                   "- Final readiness gate: pending paired bootstrap." if args.max_samples == 500 else "- Debug run only; no readiness decision.",
                   "- Final 100 accessed: No", "- Final 100 run count: 0"]
    Path(args.decision_output).write_text("\n".join(preliminary) + "\n")
    if args.max_samples == 10: print_debug(rows, [str(sample["id"]) for sample, _, _ in prepared])
    print(json.dumps({"rows": len(rows), "samples": args.max_samples, "observed_xrag_oracle_f1": observed_oracle,
                      "best_heuristic": best, "all_max_packets": max(row["num_packets"] for row in rows if row["configuration"] == "ALL"),
                      "all_max_soft_tokens": max(row["total_soft_tokens"] for row in rows if row["configuration"] == "ALL"),
                      "final_100_runs": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
