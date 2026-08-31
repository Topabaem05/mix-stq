from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-Da-d])\s*\}")
ANSWER_PATTERN = re.compile(r"\banswer\s+is\s*[:=-]?\s*\(?([A-Da-d])\)?\b", re.IGNORECASE)
TRAILING_PATTERN = re.compile(r"(?:\(([A-Da-d])\)|\b([A-Da-d])\b)\s*[.!?]?\s*$")


def parse_gold_letter(solution):
    matches = BOXED_PATTERN.findall(solution)
    return matches[-1].upper() if matches else None


def extract_prediction(text):
    boxed = list(BOXED_PATTERN.finditer(text))
    if boxed:
        return boxed[-1].group(1).upper()
    candidates = [(match.start(), match.group(1).upper()) for match in ANSWER_PATTERN.finditer(text)]
    trailing = TRAILING_PATTERN.search(text)
    if trailing:
        candidates.append((trailing.start(), (trailing.group(1) or trailing.group(2)).upper()))
    return max(candidates, default=(0, None), key=lambda candidate: candidate[0])[1]


def slice_continuations(sequences, prompt_lengths, padded_prompt_length):
    if len(sequences) != len(prompt_lengths):
        raise ValueError("one prompt length is required per generated sequence")
    continuations = []
    for sequence, prompt_length in zip(sequences, prompt_lengths, strict=True):
        if prompt_length < 0 or prompt_length > padded_prompt_length:
            raise ValueError("prompt length exceeds padded prompt width")
        unpadded = sequence[padded_prompt_length - prompt_length :]
        continuations.append(unpadded[prompt_length:])
    return continuations


def load_gpqa():
    from datasets import load_dataset

    dataset = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    items = []
    for row in dataset:
        gold = parse_gold_letter(str(row.get("solution") or ""))
        if gold is None:
            raise ValueError("GPQA row has an invalid solution")
        items.append({"problem": str(row.get("problem") or ""), "gold": gold})
    return items


def render_chat(tokenizer, problem, no_thinking):
    messages = [{"role": "user", "content": problem}]
    if no_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            pass
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_arm(model, tokenizer, items, device, generation, batch_size, no_thinking):
    import torch

    predictions = []
    batches = (len(items) + batch_size - 1) // batch_size
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [render_chat(tokenizer, item["problem"], no_thinking) for item in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_lengths = [int(length) for length in attention_mask.sum(dim=1).tolist()]
        generate_kwargs = {
            "max_new_tokens": generation["max_new_tokens"],
            "do_sample": generation["do_sample"],
            "pad_token_id": tokenizer.pad_token_id,
        }
        if generation["do_sample"]:
            generate_kwargs.update({
                "temperature": generation["temperature"],
                "top_p": generation["top_p"],
                "top_k": generation["top_k"],
            })
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        continuations = slice_continuations(sequences, prompt_lengths, input_ids.shape[1])
        texts = tokenizer.batch_decode(
            [continuation.detach().cpu().tolist() for continuation in continuations],
            skip_special_tokens=True,
        )
        predictions.extend(extract_prediction(text) for text in texts)
        batch_number = start // batch_size + 1
        print("  batch %d/%d" % (batch_number, batches), flush=True)
    return predictions


def arm_details(predictions, items):
    pairs = [
        {"predicted": prediction, "gold": item["gold"]}
        for prediction, item in zip(predictions, items, strict=True)
    ]
    correct = [int(pair["predicted"] == pair["gold"]) for pair in pairs]
    total = len(pairs)
    return correct, {
        "accuracy": sum(correct) / total if total else 0.0,
        "correct": sum(correct),
        "total": total,
        "unparsed": sum(pair["predicted"] is None for pair in pairs),
        "items": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="generative GPQA Diamond accuracy across quantization arms")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--low-layers", type=int, default=6)
    parser.add_argument("--arms", default="dense,dense_iq3")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    import torch
    from eval_mixed import apply_plan
    from eval_tasks import build_plans
    from task_accuracy import compare
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    importance = torch.load(Path(args.imatrix).with_suffix(".pt"), map_location="cpu")

    items = load_gpqa()
    print("items: %d (gpqa_diamond)" % len(items), flush=True)

    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": not args.greedy,
        "max_new_tokens": args.max_new_tokens,
    }
    config = {
        "model": args.model,
        "revision": args.revision,
        "low_layers": args.low_layers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": not args.greedy,
        "max_new_tokens": args.max_new_tokens,
        "greedy": args.greedy,
        "seed": args.seed,
        "no_thinking": args.no_thinking,
        "batch_size": args.batch_size,
    }
    plans = build_plans(args.low_layers)
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]

    results = {}
    details = {}
    cache_dir = Path(args.out).parent
    for name in selected:
        if name not in plans:
            raise ValueError("unknown arm " + name)
        cached = cache_dir / ("generative_%s.json" % name)
        if cached.is_file():
            payload = json.loads(cached.read_text(encoding="utf-8"))
            predictions = payload.get("predictions")
            if payload.get("config") == config and isinstance(predictions, list) and len(predictions) == len(items):
                correct, details[name] = arm_details(predictions, items)
                results[name] = correct
                print("[arm] %s reused from %s" % (name, cached.name), flush=True)
                continue
        print("[arm] %s" % name, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        model.eval()
        if name != "dense":
            plan_stats = apply_plan(model, importance, plans[name], device)
            if plan_stats["params"] == 0:
                raise RuntimeError(
                    "arm %s quantized nothing: expert modules did not expose fused "
                    "gate_up_proj/down_proj parameters, so the arm would silently equal dense"
                    % name
                )
            print(
                "  bpw %.4f mean_error %.4f" % (plan_stats["bpw"], plan_stats["mean_error"]),
                flush=True,
            )
        torch.manual_seed(args.seed)
        predictions = run_arm(
            model,
            tokenizer,
            items,
            device,
            generation,
            args.batch_size,
            args.no_thinking,
        )
        correct, details[name] = arm_details(predictions, items)
        results[name] = correct
        cached.write_text(
            json.dumps({"arm": name, "config": config, "predictions": predictions}),
            encoding="utf-8",
        )
        print(
            "  accuracy %.4f (%d/%d), unparsed %d"
            % (
                details[name]["accuracy"],
                details[name]["correct"],
                details[name]["total"],
                details[name]["unparsed"],
            ),
            flush=True,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = compare(results, "dense")
    for name, detail in details.items():
        report["arms"][name] = detail
    report["config"] = config
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print()
    for name, stats in report["arms"].items():
        print(
            "%-14s %.4f (%d/%d), unparsed %d"
            % (name, stats["accuracy"], stats["correct"], stats["total"], stats["unparsed"])
        )
    print()
    for label, comparison in report["comparisons"].items():
        print(
            "%-28s delta=%+.4f CI[%+.4f, %+.4f] p=%.4f %s"
            % (
                label,
                comparison["accuracy_delta"],
                comparison["ci_95"][0],
                comparison["ci_95"][1],
                comparison["mcnemar_p"],
                "SIGNIFICANT" if comparison["significant"] else "not significant",
            )
        )
    print()
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
