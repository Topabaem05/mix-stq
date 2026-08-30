from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from eval_mixed import apply_plan
from task_accuracy import compare
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]


def load_mmlu(limit, subjects=None):
    dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    items = []
    for row in dataset:
        if subjects and row.get("subject") not in subjects:
            continue
        choices = row.get("choices") or []
        if len(choices) != 4:
            continue
        items.append({
            "task": "mmlu",
            "subject": row.get("subject", ""),
            "question": (row.get("question") or "").strip(),
            "choices": [str(c).strip() for c in choices],
            "answer": int(row["answer"]),
        })
        if len(items) >= limit:
            break
    return items


def load_arc(limit):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True)
    items = []
    for row in dataset:
        choices = (row.get("choices") or {}).get("text") or []
        labels = (row.get("choices") or {}).get("label") or []
        if len(choices) != 4:
            continue
        key = row.get("answerKey")
        if key not in labels:
            continue
        items.append({
            "task": "arc_challenge",
            "subject": "arc",
            "question": (row.get("question") or "").strip(),
            "choices": [str(c).strip() for c in choices],
            "answer": labels.index(key),
        })
        if len(items) >= limit:
            break
    return items


def render(item):
    lines = [item["question"], ""]
    for letter, choice in zip(LETTERS, item["choices"], strict=True):
        lines.append("%s. %s" % (letter, choice))
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def score_item(model, tokenizer, item, device):
    prompt = render(item)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    scores = []
    with torch.no_grad():
        for letter in LETTERS:
            target = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
            if not target:
                scores.append(float("-inf"))
                continue
            target_ids = torch.tensor([target], device=device)
            ids = torch.cat([prompt_ids, target_ids], dim=1)
            logits = model(input_ids=ids).logits[0].float()
            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            span = log_probs[-target_ids.shape[1] :]
            picked = span.gather(1, target_ids[0].unsqueeze(1)).squeeze(1)
            scores.append(float(picked.sum()) / target_ids.shape[1])
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def run_arm(model, tokenizer, items, device):
    correct = []
    for item in items:
        prediction = score_item(model, tokenizer, item, device)
        correct.append(int(prediction == item["answer"]))
    return correct


def main() -> int:
    parser = argparse.ArgumentParser(description="paired task accuracy across quantization arms")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--mmlu", type=int, default=140)
    parser.add_argument("--arc", type=int, default=60)
    parser.add_argument("--low-layers", type=int, default=6)
    parser.add_argument("--arms", default="dense,mixed_stq,mixed_ltc")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    importance = torch.load(Path(args.imatrix).with_suffix(".pt"), map_location="cpu")

    items = load_mmlu(args.mmlu) + load_arc(args.arc)
    print("items: %d (mmlu %d, arc %d)" % (len(items), args.mmlu, args.arc), flush=True)

    low = set(range(args.low_layers))

    def dense(_layer, _attribute):
        return "fp16"

    def mixed_stq(layer, attribute):
        if attribute == "down_proj":
            return "iq2"
        return "stq" if layer in low else "iq2"

    def mixed_ltc(layer, attribute):
        if attribute == "down_proj":
            return "iq2"
        return "ltc" if layer in low else "iq2"

    plans = {"dense": dense, "mixed_stq": mixed_stq, "mixed_ltc": mixed_ltc}
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]

    results = {}
    details = {}
    for name in selected:
        if name not in plans:
            raise ValueError("unknown arm " + name)
        print("[arm] %s" % name, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=torch.float16
        ).to(device)
        model.eval()
        if name != "dense":
            apply_plan(model, importance, plans[name], device)
        correct = run_arm(model, tokenizer, items, device)
        results[name] = correct
        accuracy = sum(correct) / len(correct)
        details[name] = {"accuracy": accuracy, "correct": sum(correct)}
        print("  accuracy %.4f (%d/%d)" % (accuracy, sum(correct), len(correct)), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = compare(results, "dense")
    report["items_detail"] = [
        {"task": i["task"], "subject": i["subject"]} for i in items
    ]
    report["config"] = {
        "model": args.model,
        "revision": args.revision,
        "low_layers": args.low_layers,
        "mmlu": args.mmlu,
        "arc": args.arc,
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print()
    for name, stats in report["arms"].items():
        print("%-14s %.4f (%d/%d)" % (name, stats["accuracy"], stats["correct"], report["items"]))
    print()
    for label, cmp in report["comparisons"].items():
        print("%-28s delta=%+.4f CI[%+.4f, %+.4f] p=%.4f %s" % (
            label, cmp["accuracy_delta"], cmp["ci_95"][0], cmp["ci_95"][1],
            cmp["mcnemar_p"], "SIGNIFICANT" if cmp["significant"] else "not significant"))
    print()
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

