#!/usr/bin/env python
import argparse
import csv
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.language_modeling.utils import XRAG_TOKEN
from src.model import XMistralForCausalLM


VARIANTS = ["P0_CURRENT", "P1_OFFICIAL", "P2_SHORT", "P3_GROUNDED"]


def vram_gb(fn):
    return fn() / 1024**3


def parse_args():
    parser = argparse.ArgumentParser(description="HotpotQA prompt and answer calibration.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--model-name", default="Hannibal046/xrag-7b")
    parser.add_argument("--output", default="cache/results/prompt_calibration.jsonl")
    parser.add_argument("--summary-output", default="cache/results/prompt_calibration_summary.csv")
    parser.add_argument("--error-output", default="cache/results/prompt_calibration_errors.txt")
    parser.add_argument("--decision-output", default="cache/results/prompt_calibration_decision.md")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def load_hotpotqa(max_samples):
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
    )
    return dataset.select(range(min(max_samples, len(dataset))))


def load_generator(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_name)
    model = XMistralForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    print("Model:", model_name, flush=True)
    print("Tokenizer pad token id:", tokenizer.pad_token_id, flush=True)
    print("Device:", next(model.parameters()).device, flush=True)
    print("Dtype:", next(model.parameters()).dtype, flush=True)
    return tokenizer, model


def get_supporting_packets(sample):
    supporting_pairs = set(
        zip(
            sample["supporting_facts"]["title"],
            sample["supporting_facts"]["sent_id"],
        )
    )

    supporting_packets = []
    for doc_id, (title, sentences) in enumerate(
        zip(sample["context"]["title"], sample["context"]["sentences"])
    ):
        for sentence_id, sentence in enumerate(sentences):
            if (title, sentence_id) not in supporting_pairs:
                continue
            sentence = sentence.strip()
            if not sentence:
                continue
            supporting_packets.append(
                {
                    "doc_id": doc_id,
                    "sentence_id": sentence_id,
                    "title": title,
                    "text": sentence,
                }
            )

    supporting_packets.sort(key=lambda packet: (packet["doc_id"], packet["sentence_id"]))
    background_text = "\n".join(
        f"[{packet['title']}] {packet['text']}" for packet in supporting_packets
    )

    assert len(supporting_packets) >= 1
    assert background_text.strip()
    return supporting_packets, background_text


def build_prompt(variant, question, background_text):
    if variant == "P0_CURRENT":
        return (
            "Refer to the background information and answer "
            "the question with a short answer.\n\n"
            f"Background:\n{background_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if variant == "P1_OFFICIAL":
        content = (
            "Refer to the background document and answer the questions:"
            "\n\n"
            f"Background: {background_text}"
            "\n\n"
            f"Question: {question}"
        )
        return f"[INST] {content} [/INST] The answer is:"

    if variant == "P2_SHORT":
        content = (
            "Refer to the background document and answer the question. "
            "Respond only with the shortest possible answer. "
            "Do not provide an explanation."
            "\n\n"
            f"Background: {background_text}"
            "\n\n"
            f"Question: {question}"
        )
        return f"[INST] {content} [/INST] The answer is:"

    if variant == "P3_GROUNDED":
        content = (
            "Answer the question using only the supplied background. "
            "Return only the answer span, without explanation."
            "\n\n"
            f"Background: {background_text}"
            "\n\n"
            f"Question: {question}"
        )
        return f"[INST] {content} [/INST] The answer is:"

    raise ValueError(f"Unknown prompt variant: {variant}")


def assert_no_xrag_token(prompt):
    assert "[XRAG]" not in prompt
    assert "<XRAG>" not in prompt
    assert XRAG_TOKEN not in prompt


@torch.inference_mode()
def generate_raw_prediction(tokenizer, model, prompt, max_new_tokens):
    assert_no_xrag_token(prompt)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = generated[:, input_ids.shape[1]:]
    raw_prediction = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=False,
    )[0]
    assert raw_prediction is not None
    assert raw_prediction.strip()
    return raw_prediction, input_ids.shape[1], new_tokens.shape[1]


def clean_prediction(text):
    text = text.replace("</s>", " ")
    text = text.replace("<s>", " ")
    text = text.strip()

    stop_markers = [
        "\nQuestion:",
        "\nBackground:",
        "\n[INST]",
        "\n[/INST]",
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    return " ".join(text.split()).strip()


def extract_short_answer(text):
    text = clean_prediction(text)
    first_line = text.splitlines()[0].strip()
    prefixes = [
        "the answer is ",
        "answer: ",
        "it is ",
    ]
    lower = first_line.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            first_line = first_line[len(prefix) :].strip()
            break

    if "." in first_line:
        first_sentence = first_line.split(".", 1)[0].strip()
        if first_sentence:
            first_line = first_sentence

    return first_line.strip(" \t\n\"'")


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        return "".join(char for char in value if char not in string.punctuation)

    def normalize_whitespace(value):
        return " ".join(value.split())

    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    text = normalize_whitespace(text)
    return text


def exact_match(prediction, gold_answer):
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def token_f1(prediction, gold_answer):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def substring_match(prediction, gold_answer):
    gold_norm = normalize_answer(gold_answer)
    pred_norm = normalize_answer(prediction)
    return float(bool(gold_norm) and gold_norm in pred_norm)


def containment_match(prediction, gold_answer):
    gold_norm = normalize_answer(gold_answer)
    pred_norm = normalize_answer(prediction)
    return float(
        bool(gold_norm)
        and bool(pred_norm)
        and (gold_norm in pred_norm or pred_norm in gold_norm)
    )


def compute_metrics(prediction, gold_answer):
    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)
    substring = substring_match(prediction, gold_answer)
    containment = containment_match(prediction, gold_answer)

    assert em in {0.0, 1.0}
    assert 0.0 <= f1 <= 1.0
    assert substring in {0.0, 1.0}
    assert containment in {0.0, 1.0}
    return em, f1, substring, containment


def mean(values):
    return sum(values) / len(values)


def write_summary(rows, summary_output):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    summary_rows = []
    for variant in VARIANTS:
        items = grouped[variant]
        assert items, variant
        summary_rows.append(
            {
                "Variant": variant,
                "Clean EM": 100 * mean([item["clean_em"] for item in items]),
                "Clean F1": 100 * mean([item["clean_f1"] for item in items]),
                "Clean substring": 100 * mean([item["clean_substring"] for item in items]),
                "Clean containment": 100 * mean([item["clean_containment"] for item in items]),
                "Short EM": 100 * mean([item["short_em"] for item in items]),
                "Short F1": 100 * mean([item["short_f1"] for item in items]),
                "Short substring": 100 * mean([item["short_substring"] for item in items]),
                "Short containment": 100 * mean([item["short_containment"] for item in items]),
                "Avg input tokens": mean([item["input_tokens"] for item in items]),
                "Avg generated tokens": mean([item["generated_tokens"] for item in items]),
            }
        )

    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Variant",
        "Clean EM",
        "Clean F1",
        "Clean substring",
        "Clean containment",
        "Short EM",
        "Short F1",
        "Short substring",
        "Short containment",
        "Avg input tokens",
        "Avg generated tokens",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSummary", flush=True)
    print(
        "| Variant | Clean EM | Clean F1 | Clean substring | Clean containment | "
        "Short EM | Short F1 | Short substring | Short containment | Avg input tokens | Avg generated tokens |",
        flush=True,
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for row in summary_rows:
        print(
            f"| {row['Variant']} | {row['Clean EM']:.2f} | {row['Clean F1']:.2f} | "
            f"{row['Clean substring']:.2f} | {row['Clean containment']:.2f} | "
            f"{row['Short EM']:.2f} | {row['Short F1']:.2f} | "
            f"{row['Short substring']:.2f} | {row['Short containment']:.2f} | "
            f"{row['Avg input tokens']:.2f} | {row['Avg generated tokens']:.2f} |",
            flush=True,
        )
    return summary_rows


def best_row(summary_rows, metric):
    return max(summary_rows, key=lambda row: row[metric])


def format_example(row):
    metrics = (
        f"clean_em={row['clean_em']:.0f} clean_f1={row['clean_f1']:.3f} "
        f"clean_substring={row['clean_substring']:.0f} clean_containment={row['clean_containment']:.0f} "
        f"short_em={row['short_em']:.0f} short_f1={row['short_f1']:.3f} "
        f"short_substring={row['short_substring']:.0f} short_containment={row['short_containment']:.0f}"
    )
    return (
        f"Variant: {row['variant']}\n"
        f"Sample: {row['sample_id']}\n"
        f"Question: {row['question']}\n"
        f"Gold answer: {row['gold_answer']}\n"
        f"Supporting text:\n{row['supporting_text']}\n"
        f"Raw prediction: {row['raw_prediction']}\n"
        f"Clean prediction: {row['clean_prediction']}\n"
        f"Short prediction: {row['short_prediction']}\n"
        f"Metrics: {metrics}\n"
    )


def write_error_analysis(rows, summary_rows, error_output):
    best_variant = best_row(summary_rows, "Clean F1")["Variant"]
    best_items = [row for row in rows if row["variant"] == best_variant]

    substring_not_em = [
        row for row in best_items if row["clean_substring"] == 1.0 and row["clean_em"] == 0.0
    ][:10]
    fully_wrong = [
        row for row in best_items if row["clean_f1"] == 0.0 and row["clean_substring"] == 0.0
    ][:10]
    short_helped = [
        row for row in best_items if row["short_f1"] - row["clean_f1"] >= 0.3
    ][:10]
    selected_ids = {
        (row["sample_id"], row["variant"], row["raw_prediction"])
        for row in substring_not_em + fully_wrong + short_helped
    }
    needed = max(0, 30 - len(selected_ids))
    supplemental = []
    if needed:
        candidates = sorted(
            best_items,
            key=lambda row: (row["clean_f1"], row["clean_em"], -row["clean_substring"]),
        )
        for row in candidates:
            row_id = (row["sample_id"], row["variant"], row["raw_prediction"])
            if row_id in selected_ids:
                continue
            supplemental.append(row)
            selected_ids.add(row_id)
            if len(supplemental) == needed:
                break

    output = [
        "# Prompt Calibration Error Analysis",
        "",
        f"Best variant by Clean F1: {best_variant}",
        "",
        "## A. Substring Correct But EM Wrong",
        "",
    ]
    output.extend(format_example(row) for row in substring_not_em)
    output.extend(
        [
            f"Count shown: {len(substring_not_em)}",
            "",
            "## B. Completely Wrong",
            "",
        ]
    )
    output.extend(format_example(row) for row in fully_wrong)
    output.extend(
        [
            f"Count shown: {len(fully_wrong)}",
            "",
            "## C. Short Extraction Helped",
            "",
        ]
    )
    output.extend(format_example(row) for row in short_helped)
    output.extend(
        [
            f"Count shown: {len(short_helped)}",
            "",
            "## D. Supplemental Low-F1 Examples",
            "",
        ]
    )
    output.extend(format_example(row) for row in supplemental)
    output.extend(
        [
            f"Count shown: {len(supplemental)}",
            "",
            "## Aggregate Observations",
            "",
            "- Substring-correct/EM-wrong examples indicate whether extra explanation is hurting exact match.",
            "- Fully wrong examples show whether the generator ignores evidence or fails multi-hop composition.",
            "- Short-helped examples estimate how much conservative answer extraction can recover.",
            "- Supplemental examples are included only when the strict three diagnostic buckets contain fewer than 30 total records.",
        ]
    )

    error_path = Path(error_output)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text("\n".join(output))
    return {
        "best_variant": best_variant,
        "substring_not_em_count": len(substring_not_em),
        "fully_wrong_count": len(fully_wrong),
        "short_helped_count": len(short_helped),
        "supplemental_count": len(supplemental),
        "total_examples": len(substring_not_em) + len(fully_wrong) + len(short_helped) + len(supplemental),
    }


def build_decision(summary_rows, error_info):
    best_clean = best_row(summary_rows, "Clean F1")
    best_short = best_row(summary_rows, "Short F1")
    best_substring = best_row(summary_rows, "Clean substring")
    p_rows = {row["Variant"]: row for row in summary_rows}

    short_clean_gap = best_short["Short F1"] - best_clean["Clean F1"]
    substring_em_gap = best_substring["Clean substring"] - best_clean["Clean EM"]

    path1 = best_short["Short F1"] >= 50.0 or best_substring["Clean substring"] >= 60.0
    path2 = best_substring["Clean substring"] >= 60.0 and best_short["Short F1"] < 45.0
    path3 = best_short["Short F1"] < 45.0 and best_substring["Clean substring"] < 50.0

    if path2:
        final_path = "Path 2: metric/extraction is the main issue; improve answer-span extraction."
    elif path1:
        final_path = "Path 1: prompt/extraction fix is promising; fix best prompt/extraction and rerun selector diagnostic."
    elif path3:
        final_path = "Path 3: generator remains weak; evaluate stronger text generator upper bounds."
    else:
        final_path = "Path 4: inspect data/evaluation implementation for remaining mismatch."

    lines = [
        "# Prompt Calibration Decision",
        "",
        "## 1. Prompt Variant Results",
        "",
        "| Variant | Clean F1 | Short F1 | Clean substring | Short containment |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['Variant']} | {row['Clean F1']:.2f} | {row['Short F1']:.2f} | "
            f"{row['Clean substring']:.2f} | {row['Short containment']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Clean vs Short",
            "",
            f"Best variant by Clean F1: {best_clean['Variant']} ({best_clean['Clean F1']:.2f})",
            f"Best variant by Short F1: {best_short['Variant']} ({best_short['Short F1']:.2f})",
            f"Short F1 - Clean F1 gap: {short_clean_gap:.2f}",
            "",
            "## 3. EM/F1 vs Substring",
            "",
            f"Best variant by substring match: {best_substring['Variant']} ({best_substring['Clean substring']:.2f})",
            f"Substring - EM gap: {substring_em_gap:.2f}",
            "",
            "## 4. Error Analysis Coverage",
            "",
            f"Best error-analysis variant: {error_info['best_variant']}",
            f"Substring-correct/EM-wrong examples: {error_info['substring_not_em_count']}",
            f"Completely wrong examples: {error_info['fully_wrong_count']}",
            f"Short-extraction-helped examples: {error_info['short_helped_count']}",
            f"Supplemental low-F1 examples: {error_info['supplemental_count']}",
            f"Total error-analysis examples: {error_info['total_examples']}",
            "",
            "Conclusion: the error file contains at least 30 sampled examples for manual inspection. "
            "Supplemental low-F1 examples fill any shortage from the strict diagnostic buckets.",
            "",
            f"## 5. Final Path",
            "",
            final_path,
            "",
            "Triggers:",
            f"- Path 1 trigger: {path1}",
            f"- Path 2 trigger: {path2}",
            f"- Path 3 trigger: {path3}",
            f"- Path 4 fallback: {not (path1 or path2 or path3)}",
        ]
    )

    comparisons = {
        "best_clean_variant": best_clean["Variant"],
        "best_short_variant": best_short["Variant"],
        "best_substring_variant": best_substring["Variant"],
        "best_clean_f1": best_clean["Clean F1"],
        "best_short_f1": best_short["Short F1"],
        "best_substring": best_substring["Clean substring"],
        "short_clean_gap": short_clean_gap,
        "substring_em_gap": substring_em_gap,
        "p0_clean_f1": p_rows["P0_CURRENT"]["Clean F1"],
        "p1_clean_f1": p_rows["P1_OFFICIAL"]["Clean F1"],
        "p2_clean_f1": p_rows["P2_SHORT"]["Clean F1"],
        "p3_clean_f1": p_rows["P3_GROUNDED"]["Clean F1"],
        "final_path": final_path,
    }
    return "\n".join(lines) + "\n", comparisons


def print_final_results(comparisons):
    print("", flush=True)
    print(f"Best variant by Clean F1: {comparisons['best_clean_variant']}", flush=True)
    print(f"Best variant by Short F1: {comparisons['best_short_variant']}", flush=True)
    print(f"Best variant by Substring Match: {comparisons['best_substring_variant']}", flush=True)
    print("", flush=True)
    print(f"P0 Clean F1: {comparisons['p0_clean_f1']:.2f}", flush=True)
    print(f"P1 Clean F1: {comparisons['p1_clean_f1']:.2f}", flush=True)
    print(f"P2 Clean F1: {comparisons['p2_clean_f1']:.2f}", flush=True)
    print(f"P3 Clean F1: {comparisons['p3_clean_f1']:.2f}", flush=True)
    print("", flush=True)
    print(f"Best Clean F1: {comparisons['best_clean_f1']:.2f}", flush=True)
    print(f"Best Short F1: {comparisons['best_short_f1']:.2f}", flush=True)
    print(f"Best Substring Match: {comparisons['best_substring']:.2f}", flush=True)
    print("", flush=True)
    print(f"Short F1 - Clean F1 gap: {comparisons['short_clean_gap']:.2f}", flush=True)
    print(f"Substring - EM gap: {comparisons['substring_em_gap']:.2f}", flush=True)
    print(f"Decision: {comparisons['final_path']}", flush=True)


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer, model = load_generator(args.model_name, device)

    print(f"Loading HotpotQA distractor validation max_samples={args.max_samples}", flush=True)
    dataset = load_hotpotqa(args.max_samples)
    print(f"Loaded samples: {len(dataset)}", flush=True)

    rows = []
    start = time.perf_counter()
    with output_path.open("w") as out:
        for sample_index, sample in enumerate(dataset):
            sample_id = str(sample.get("id", sample.get("_id", sample_index)))
            question = sample["question"]
            gold_answer = sample["answer"]
            supporting_packets, background_text = get_supporting_packets(sample)

            print(
                f"[{sample_index + 1}/{len(dataset)}] {sample_id} "
                f"supporting={len(supporting_packets)}",
                flush=True,
            )

            for variant in VARIANTS:
                prompt = build_prompt(variant, question, background_text)
                assert_no_xrag_token(prompt)
                raw_prediction, input_tokens, generated_tokens = generate_raw_prediction(
                    tokenizer,
                    model,
                    prompt,
                    args.max_new_tokens,
                )
                clean = clean_prediction(raw_prediction)
                short = extract_short_answer(raw_prediction)
                assert clean.strip()
                assert short.strip()

                clean_em, clean_f1, clean_substring, clean_containment = compute_metrics(clean, gold_answer)
                short_em, short_f1, short_substring, short_containment = compute_metrics(short, gold_answer)

                row = {
                    "sample_id": sample_id,
                    "variant": variant,
                    "question": question,
                    "gold_answer": gold_answer,
                    "supporting_text": background_text,
                    "num_supporting_sentences": len(supporting_packets),
                    "prompt": prompt,
                    "raw_prediction": raw_prediction,
                    "clean_prediction": clean,
                    "short_prediction": short,
                    "clean_em": clean_em,
                    "clean_f1": clean_f1,
                    "clean_substring": clean_substring,
                    "clean_containment": clean_containment,
                    "short_em": short_em,
                    "short_f1": short_f1,
                    "short_substring": short_substring,
                    "short_containment": short_containment,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(row)
                print(
                    f"  {variant} clean_f1={clean_f1:.3f} short_f1={short_f1:.3f} "
                    f"sub={clean_substring:.0f} in={input_tokens} gen={generated_tokens} "
                    f"raw={raw_prediction[:80]!r}",
                    flush=True,
                )

    total_rows = len(rows)
    assert total_rows == len(dataset) * 4
    print(f"\nTotal runtime seconds: {time.perf_counter() - start:.3f}", flush=True)
    print(f"Peak allocated GB: {vram_gb(torch.cuda.max_memory_allocated):.3f}", flush=True)
    print(f"Peak reserved GB: {vram_gb(torch.cuda.max_memory_reserved):.3f}", flush=True)
    print(f"Rows written: {total_rows}", flush=True)
    print(f"Predictions written to: {output_path}", flush=True)

    summary_rows = write_summary(rows, args.summary_output)
    print(f"Summary written to: {args.summary_output}", flush=True)

    error_info = write_error_analysis(rows, summary_rows, args.error_output)
    print(f"Error analysis written to: {args.error_output}", flush=True)

    decision_md, comparisons = build_decision(summary_rows, error_info)
    decision_path = Path(args.decision_output)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(decision_md)
    print_final_results(comparisons)
    print(f"Decision written to: {decision_path}", flush=True)


if __name__ == "__main__":
    main()
