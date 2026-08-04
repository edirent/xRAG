#!/usr/bin/env python
"""Run the locked no-training composition-gap diagnostic suite on DEV-150."""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector
from scripts.packet_xrag import train_packet_projector as v1
from scripts.packet_xrag.run_static_scorer_benchmark import initialize_generator
from scripts.packet_xrag.utility_predictor_training_common import load_static_score_cache
from src.language_modeling.utils import XRAG_TOKEN
from src.model import SFR, XMistralForCausalLM
from src.model.xMistral.modeling_xmistral import Projector
from src.packet_xrag.controller.feature_cache import ControllerFeatureCache
from src.packet_xrag.encoding.multi_token_projector import MultiTokenPacketProjector
from src.packet_xrag.modeling.multi_token_xrag import install_multi_token_injection


K4_SHA256 = "971df4f4c516dc8945b2a5e2ba9f80f69279ac3cc0691635beeff9e21498c7d4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cache/composition")
    parser.add_argument("--feature-cache", default="cache/controller/features/train_features")
    parser.add_argument("--static-checkpoint", default="cache/controller/static/best_short_f1/scorer.pt")
    parser.add_argument("--static-score-cache", default="cache/controller/utility_predictor/features/train_static_scores.pt")
    parser.add_argument("--k2-training-config", default="cache/projector/multi_token_k2/best_short_f1/training_config.json")
    parser.add_argument("--k4-checkpoint", default="cache/projector/multi_token_k4/best_short_f1/multi_token_projector.pt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def token_statistics(tokenizer, scores, generated, index, length):
    logps, entropies = [], []
    for step in range(length):
        log_probs = F.log_softmax(scores[step][index].float(), -1)
        logps.append(float(log_probs[int(generated[index, step])]))
        probabilities = log_probs.exp(); entropies.append(float(-(probabilities * log_probs).sum()))
    return mean(logps), min(logps), mean(entropies)


@torch.inference_mode()
def generate_retrieval_batch(tokenizer, model, xrag_id, questions, retrieval_groups,
                             placeholders_per_group, device, max_new_tokens=32):
    prompts = [v1.build_prompt(question, count) for question, count in
               zip(questions, placeholders_per_group)]
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                       padding=True).to(device)
    retrieval = torch.cat([value.to(device) for value in retrieval_groups], 0)
    if int(inputs.input_ids.eq(xrag_id).sum()) != sum(placeholders_per_group):
        raise RuntimeError("retrieval diagnostic XRAG placeholder mismatch")
    output = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        retrieval_embeds=retrieval, do_sample=False, max_new_tokens=max_new_tokens,
        use_cache=True, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=True)
    generated = output.sequences[:, -len(output.scores):]
    rows = []
    for index in range(len(questions)):
        eos = generated[index].eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
        length = int(eos[0]) + 1 if len(eos) else generated.shape[1]
        raw = tokenizer.decode(generated[index, :length], skip_special_tokens=False)
        confidence, minimum, entropy = token_statistics(tokenizer, output.scores, generated,
                                                         index, length)
        rows.append({"short_prediction": selector.extract_short_answer(raw) or "[EMPTY]",
                     "raw_generation": raw, "prompt_tokens": int(inputs.attention_mask[index].sum()),
                     "generated_tokens": length, "mean_token_logprob": confidence,
                     "minimum_token_logprob": minimum, "mean_token_entropy": entropy})
    return rows


def text_prompt(question, packets):
    background = "\n".join(packet["encoder_text"] for packet in packets)
    content = ("Refer to the background document and answer the question. "
               "Respond only with the shortest possible answer. Do not provide an explanation."
               f"\n\nBackground: {background}\n\nQuestion: {question}")
    return f"[INST] {content} [/INST] The answer is:"


@torch.inference_mode()
def generate_text_batch(tokenizer, model, questions, packet_groups, device):
    prompts = [text_prompt(question, packets) for question, packets in zip(questions, packet_groups)]
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
    output = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        do_sample=False, max_new_tokens=32, use_cache=True, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=True)
    generated = output.sequences[:, -len(output.scores):]
    rows = []
    for index, packets in enumerate(packet_groups):
        eos = generated[index].eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
        length = int(eos[0]) + 1 if len(eos) else generated.shape[1]
        raw = tokenizer.decode(generated[index, :length], skip_special_tokens=False)
        confidence, minimum, entropy = token_statistics(tokenizer, output.scores, generated,
                                                         index, length)
        background = "\n".join(packet["encoder_text"] for packet in packets)
        text_tokens = len(tokenizer(background, add_special_tokens=False).input_ids)
        rows.append({"short_prediction": selector.extract_short_answer(raw) or "[EMPTY]",
                     "raw_generation": raw, "prompt_tokens": int(inputs.attention_mask[index].sum()),
                     "text_tokens": text_tokens, "generated_tokens": length,
                     "mean_token_logprob": confidence, "minimum_token_logprob": minimum,
                     "mean_token_entropy": entropy})
    return rows


def result_row(record, configuration, selected, generated, soft_tokens, text_tokens=0):
    prediction = generated["short_prediction"]
    em, f1 = selector.score_prediction(prediction, record["answer"])
    gold, chosen = set(record["gold_packet_ids"]), set(selected)
    return {"sample_id": record["sample_id"], "configuration": configuration,
            "selected_packet_ids": list(selected), "short_prediction": prediction,
            "short_em": em, "short_f1": f1, "is_empty": prediction == "[EMPTY]",
            "num_packets": len(selected), "soft_tokens": soft_tokens,
            "text_tokens": text_tokens, "prompt_tokens": generated["prompt_tokens"],
            "generated_tokens": generated["generated_tokens"],
            "mean_token_logprob": generated["mean_token_logprob"],
            "minimum_token_logprob": generated["minimum_token_logprob"],
            "mean_token_entropy": generated["mean_token_entropy"],
            "support_recall": len(gold & chosen) / len(gold),
            "full_support": float(gold.issubset(chosen))}


def load_k4_model(device, tokenizer, xrag_id, v1_path, k4_path):
    import hashlib
    if hashlib.sha256(Path(k4_path).read_bytes()).hexdigest() != K4_SHA256:
        raise RuntimeError("formal K4 checkpoint hash mismatch")
    config = AutoConfig.from_pretrained(v1.XRAG_MODEL_NAME)
    model = XMistralForCausalLM.from_pretrained(v1.XRAG_MODEL_NAME, config=config,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.set_xrag_token_id(xrag_id)
    model.projector.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True), strict=True)
    model.projector = MultiTokenPacketProjector(model.projector, config.retriever_hidden_size,
                                                 config.hidden_size, 4, 1024).to(
                                                     device=device, dtype=torch.bfloat16)
    model.projector.load_state_dict(torch.load(k4_path, map_location="cpu", weights_only=True), strict=True)
    install_multi_token_injection(model); model.eval()
    for parameter in model.parameters(): parameter.requires_grad = False
    return model


def main(argv=None):
    args = parse_args(argv); root = Path(args.root); output_dir = root / "diagnostics"
    output = output_dir / "diagnostic_predictions.jsonl"
    if output.exists(): raise RuntimeError("refusing to overwrite composition diagnostic")
    ids = json.loads((output_dir / "diagnostic_ids.json").read_text())["ordered_sample_ids"]
    stress = {row["sample_id"]: row for row in
              [json.loads(line) for line in (output_dir / "stress_sets.jsonl").read_text().splitlines()]}
    parent = ControllerFeatureCache(args.feature_cache)
    by_id = {record["sample_id"]: index for index, record in enumerate(parent.records)}
    records = [parent[by_id[sid]] for sid in ids]
    device = torch.device(args.device); torch.cuda.set_device(device)
    static_scores = load_static_score_cache(parent, args.static_checkpoint,
                                             args.static_score_cache, device)
    tokenizer, generator, xrag_id, hashes = initialize_generator(args.k2_training_config, device)
    rows, timings = [], defaultdict(float)

    def evaluate_soft(configuration, selected_groups, timer_group):
        began = time.time()
        for start in range(0, len(records), args.batch_size):
            current_records = records[start:start + args.batch_size]
            selected = selected_groups[start:start + args.batch_size]
            generated = generate_retrieval_batch(tokenizer, generator, xrag_id,
                [record["question"] for record in current_records],
                [record["packet_embeddings"][packet_ids] for record, packet_ids in zip(current_records, selected)],
                [2 * len(packet_ids) for packet_ids in selected], device)
            rows.extend(result_row(record, configuration, packet_ids, result,
                                   2 * len(packet_ids)) for record, packet_ids, result in
                        zip(current_records, selected, generated))
        timings[timer_group] += time.time() - began
        print(json.dumps({"configuration": configuration, "samples": len(records)}), flush=True)

    rankings = {record["sample_id"]: sorted(range(record["packet_count"]),
                key=lambda index: (-float(static_scores[record["sample_id"]][index]), index))
                for record in records}
    evaluate_soft("STATIC_2", [rankings[r["sample_id"]][:2] for r in records], "breadth")
    for breadth in (2, 3, 4, 6, 12):
        evaluate_soft(f"TOPK_{breadth}", [r["topk_ranking"][:breadth] for r in records], "breadth")
    evaluate_soft("ALL", [r["topk_ranking"] for r in records], "breadth")

    for breadth in (4, 6):
        for variant in ("reverse", "document", "random_0", "random_1", "random_2"):
            name = f"ORDER_TOPK{breadth}_{variant.upper()}"
            evaluate_soft(name, [stress[r["sample_id"]]["orders"][str(breadth)][variant]
                                 for r in records], "order")
    for configuration in ("DUP_GOLD_X1", "DUP_GOLD_X2", "DUP_GOLD_X4",
                          "DUP_NONGOLD_X1", "DUP_NONGOLD_X2", "DUP_NONGOLD_X4",
                          "DUP_RANDOM_X1", "DUP_RANDOM_X2", "DUP_RANDOM_X4"):
        evaluate_soft(configuration, [stress[r["sample_id"]]["duplicates"][configuration]
                                      for r in records], "duplicate")

    # Original-text controls use exactly the same TOPK packet identities.
    began = time.time()
    for breadth in (2, 4, 6, None):
        configuration = "TEXT_ALL" if breadth is None else f"TEXT_TOP{breadth}"
        selected_groups = [r["topk_ranking"] if breadth is None else r["topk_ranking"][:breadth]
                           for r in records]
        for start in range(0, len(records), 4):
            current_records = records[start:start + 4]; selected = selected_groups[start:start + 4]
            packet_groups = [[record["packets"][index] for index in packet_ids]
                             for record, packet_ids in zip(current_records, selected)]
            generated = generate_text_batch(tokenizer, generator,
                [record["question"] for record in current_records], packet_groups, device)
            rows.extend(result_row(record, configuration, packet_ids, result, 0,
                                   result["text_tokens"]) for record, packet_ids, result in
                        zip(current_records, selected, generated))
        print(json.dumps({"configuration": configuration, "samples": len(records)}), flush=True)
    timings["text"] = time.time() - began

    # Group multiple packet texts into one OOD SFR embedding, then apply frozen K2.
    began = time.time(); sfr_tokenizer = AutoTokenizer.from_pretrained(v1.SFR_MODEL_NAME)
    sfr = SFR.from_pretrained(v1.SFR_MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    grouped_embeddings = {}
    for breadth in (2, 4, 6):
        values = []
        texts = ["\n".join(r["packets"][index]["encoder_text"]
                            for index in r["topk_ranking"][:breadth]) for r in records]
        for start in range(0, len(texts), 8):
            inputs = sfr_tokenizer(texts[start:start + 8], max_length=512, padding=True,
                                   truncation=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                embedded = sfr.get_doc_embedding(input_ids=inputs.input_ids,
                                                   attention_mask=inputs.attention_mask)
            values.extend(value.detach().cpu() for value in embedded.view(-1, embedded.shape[-1]))
        grouped_embeddings[breadth] = values
    del sfr, sfr_tokenizer; gc.collect(); torch.cuda.empty_cache()
    timings["grouped_encoding"] = time.time() - began
    for breadth in (2, 4, 6):
        configuration = f"GROUPED_TOP{breadth}_K2"; began = time.time()
        for start in range(0, len(records), args.batch_size):
            current = records[start:start + args.batch_size]
            embeddings = [value.unsqueeze(0) for value in grouped_embeddings[breadth][start:start + args.batch_size]]
            generated = generate_retrieval_batch(tokenizer, generator, xrag_id,
                [record["question"] for record in current], embeddings, [2] * len(current), device)
            for record, result in zip(current, generated):
                selected = record["topk_ranking"][:breadth]
                rows.append(result_row(record, configuration, selected, result, 2))
        timings["grouped"] += time.time() - began
        print(json.dumps({"configuration": configuration, "samples": len(records)}), flush=True)

    # Compatible formal K4 provides the 4-token grouped-vs-independent comparison.
    k2_config = json.loads(Path(args.k2_training_config).read_text()); v1_path = k2_config["base_projector_checkpoint"]
    del generator; gc.collect(); torch.cuda.empty_cache()
    generator = load_k4_model(device, tokenizer, xrag_id, v1_path, args.k4_checkpoint)
    began = time.time(); configuration = "GROUPED_TOP2_K4"
    for start in range(0, len(records), args.batch_size):
        current = records[start:start + args.batch_size]
        embeddings = [value.unsqueeze(0) for value in grouped_embeddings[2][start:start + args.batch_size]]
        generated = generate_retrieval_batch(tokenizer, generator, xrag_id,
            [record["question"] for record in current], embeddings, [4] * len(current), device)
        for record, result in zip(current, generated):
            rows.append(result_row(record, configuration, record["topk_ranking"][:2], result, 4))
    timings["same_bandwidth_k4"] = time.time() - began
    print(json.dumps({"configuration": configuration, "samples": len(records)}), flush=True)

    with output.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"status": "complete", "split": "COMPOSITION_DEV_DIAGNOSTIC_150",
                "sample_count": len(records), "configuration_count": len(set(r["configuration"] for r in rows)),
                "prediction_count": len(rows), "checkpoint_hashes": {**hashes, "k4": K4_SHA256},
                "k1_same_bandwidth": "SKIPPED_NO_FORMAL_COMPATIBLE_K1_CHECKPOINT",
                "k4_same_bandwidth": "GROUPED_TOP2_K4", "timings_seconds": timings,
                "generator_trainable_parameters": 0, "sfr_trainable_parameters": 0,
                "search_shadow_accessed": False, "benchmark_accessed": False,
                "final_100_accessed": False}
    (output_dir / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                                     sort_keys=True) + "\n")
    ledger_path = root / "experiment_ledger.json"; ledger = json.loads(ledger_path.read_text())
    ledger["usage"]["diagnostic_probe_runs"] = 5
    ledger["usage"]["composition_dev_generation"] = 5
    ledger["stage1_diagnostic_manifest"] = manifest
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
