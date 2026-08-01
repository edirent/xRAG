#!/usr/bin/env python
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.packet_xrag import run_selector_calibration as selector


METHODS = ["TEXT_ORACLE", "XRAG_ORACLE", "ORACLE_1", "ORACLE_2", "ALL"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the calibrated projector gate.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--projector-checkpoint",
        default="cache/projector/packet_projector_calibration/last/projector.pt",
    )
    parser.add_argument("--output", default="cache/results/projector_gate.jsonl")
    parser.add_argument(
        "--summary-output", default="cache/results/projector_gate_summary.csv"
    )
    parser.add_argument(
        "--decision-output", default="cache/results/projector_gate_decision.md"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def write_summary(rows, output):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    budgets = {
        "TEXT_ORACLE": "",
        "XRAG_ORACLE": "",
        "ORACLE_1": 1,
        "ORACLE_2": 2,
        "ALL": "",
    }
    summary = []
    for method in METHODS:
        items = groups[method]
        summary.append(
            {
                "Method": method,
                "Budget": budgets[method],
                "Short EM": 100 * mean([row["short_em"] for row in items]),
                "Short F1": 100 * mean([row["short_f1"] for row in items]),
                "Clean F1": 100 * mean([row["clean_f1"] for row in items]),
                "Avg packets": mean([row["num_packets"] for row in items]),
                "Support recall": mean([row["support_recall"] for row in items]),
                "Full support": mean(
                    [row["full_support_coverage"] for row in items]
                ),
            }
        )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def build_decision(summary):
    by_method = {row["Method"]: row for row in summary}
    text = by_method["TEXT_ORACLE"]["Short F1"]
    xrag = by_method["XRAG_ORACLE"]["Short F1"]
    oracle_1 = by_method["ORACLE_1"]["Short F1"]
    oracle_2 = by_method["ORACLE_2"]["Short F1"]
    all_f1 = by_method["ALL"]["Short F1"]
    gap = text - xrag
    minimum_gate = xrag >= 55.0
    ideal_gate = gap <= 8.0
    oracle_2_close = abs(oracle_2 - xrag) <= 5.0
    multihop_signal = oracle_2 - oracle_1 >= 5.0
    overload_preserved = xrag - all_f1 >= 5.0
    passed = minimum_gate
    recommendation = (
        "Projector gate passed. Run the complete selector calibration."
        if passed
        else "Projector gate failed. Do not start controller training."
    )
    lines = [
        "# Packet Projector Gate",
        "",
        f"- TEXT_ORACLE Short F1: {text:.2f}",
        f"- XRAG_ORACLE Short F1: {xrag:.2f}",
        f"- ORACLE_1 Short F1: {oracle_1:.2f}",
        f"- ORACLE_2 Short F1: {oracle_2:.2f}",
        f"- ALL Short F1: {all_f1:.2f}",
        f"- Representation gap: {gap:.2f}",
        "",
        f"- Minimum gate (XRAG_ORACLE >= 55): {minimum_gate}",
        f"- Ideal gate (gap <= 8): {ideal_gate}",
        f"- ORACLE_2 close to XRAG_ORACLE: {oracle_2_close}",
        f"- ORACLE_2 clearly above ORACLE_1: {multihop_signal}",
        f"- Sparse-oracle overload motivation preserved: {overload_preserved}",
        "",
        f"Decision: {recommendation}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sfr_tokenizer, sfr_model, tokenizer, model, xrag_token_id = selector.load_models(
        device, torch.bfloat16
    )
    state = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.projector.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    model.eval()
    print(f"Loaded projector: {args.projector_checkpoint}", flush=True)

    dataset = selector.load_hotpotqa(args.max_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    with output_path.open("w") as handle:
        for sample_index, sample in enumerate(dataset):
            sample_id = str(sample.get("id", sample.get("_id", sample_index)))
            question, gold_answer = sample["question"], sample["answer"]
            packets = selector.make_packets(sample)
            gold_indices = selector.select_oracle(packets)
            _, packet_raw, _, _ = selector.encode_query_and_packets(
                sfr_tokenizer, sfr_model, question, packets, device
            )
            runs = [
                ("TEXT_ORACLE", None, gold_indices, "text"),
                ("XRAG_ORACLE", None, gold_indices, "xrag"),
                ("ORACLE_1", 1, gold_indices[:1], "xrag"),
                ("ORACLE_2", 2, gold_indices[:2], "xrag"),
                ("ALL", None, list(range(len(packets))), "xrag"),
            ]
            for method, budget, selected, mode in runs:
                if mode == "text":
                    prompt = selector.build_text_oracle_prompt(
                        question, packets, selected
                    )
                    raw, input_tokens, generated_tokens = selector.generate_text_answer(
                        tokenizer, model, prompt, device, args.max_new_tokens
                    )
                else:
                    prompt = selector.build_xrag_prompt(question, len(selected))
                    raw, input_tokens, generated_tokens = selector.generate_xrag_answer(
                        tokenizer,
                        model,
                        xrag_token_id,
                        question,
                        packet_raw[selected],
                        device,
                        args.max_new_tokens,
                    )
                clean = selector.clean_prediction(raw)
                short = selector.extract_short_answer(raw)
                clean_em, clean_f1 = selector.score_prediction(clean, gold_answer)
                short_em, short_f1 = selector.score_prediction(short, gold_answer)
                _, recall, precision, full = selector.coverage_metrics(
                    selected, set(gold_indices)
                )
                row = {
                    "sample_id": sample_id,
                    "method": method,
                    "budget": budget,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "raw_prediction": raw,
                    "clean_prediction": clean,
                    "short_prediction": short,
                    "clean_em": clean_em,
                    "clean_f1": clean_f1,
                    "short_em": short_em,
                    "short_f1": short_f1,
                    "num_packets": len(selected),
                    "total_packets": len(packets),
                    "selected_indices": selected,
                    "gold_supporting_indices": gold_indices,
                    "support_recall": recall,
                    "support_precision": precision,
                    "full_support_coverage": full,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
            print(f"[{sample_index + 1}/{len(dataset)}] {sample_id}", flush=True)

    assert len(rows) == len(dataset) * len(METHODS)
    summary = write_summary(rows, args.summary_output)
    decision = build_decision(summary)
    Path(args.decision_output).write_text(decision)
    print(Path(args.summary_output).read_text(), flush=True)
    print(decision, flush=True)
    print(f"Runtime seconds: {time.perf_counter() - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
