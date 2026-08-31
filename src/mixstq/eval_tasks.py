from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from eval_mixed import apply_plan
from task_accuracy import compare
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]
MMLU_DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
ARC_DATASET_REVISION = "210d026faf9955653af8916fad021475a3f00453"
MMLU_SUBJECT_COUNT = 57


def item_fingerprint(items):
    canonical = json.dumps(
        items, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_provenance(
    model, revision, dtype, mmlu, arc, low_layers, arm, sampling_scheme, fingerprint
):
    return {
        "model": model,
        "revision": revision,
        "dtype": dtype,
        "mmlu": mmlu,
        "arc": arc,
        "low_layers": low_layers,
        "arm": arm,
        "sampling_scheme": sampling_scheme,
        "dataset_revisions": {
            "cais/mmlu": MMLU_DATASET_REVISION,
            "allenai/ai2_arc": ARC_DATASET_REVISION,
        },
        "ordered_item_fingerprint": fingerprint,
    }


def load_cached_correct(path, provenance, item_count):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    correct = payload.get("correct")
    if payload.get("provenance") != provenance:
        return None
    if not isinstance(correct, list) or len(correct) != item_count:
        return None
    if any(type(value) is not int or value not in (0, 1) for value in correct):
        return None
    return correct


def single_token_letters(tokenizer):
    tokens = []
    for letter in LETTERS:
        ids = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            return None
        tokens.append(int(ids[0]))
    return tokens


def mmlu_item(row):
    choices = row.get("choices") or []
    if len(choices) != 4:
        return None
    return {
        "task": "mmlu",
        "subject": row.get("subject", ""),
        "question": (row.get("question") or "").strip(),
        "choices": [str(c).strip() for c in choices],
        "answer": int(row["answer"]),
    }


def load_mmlu(limit, subjects=None):
    dataset = load_dataset(
        "cais/mmlu",
        "all",
        split="test",
        streaming=True,
        revision=MMLU_DATASET_REVISION,
    )
    items = []
    for row in dataset:
        if subjects and row.get("subject") not in subjects:
            continue
        item = mmlu_item(row)
        if item is None:
            continue
        items.append(item)
        if len(items) >= limit:
            break
    return items


def load_mmlu_stratified(per_subject):
    dataset = load_dataset(
        "cais/mmlu",
        "all",
        split="test",
        streaming=True,
        revision=MMLU_DATASET_REVISION,
    )
    by_subject = {}
    for row in dataset:
        item = mmlu_item(row)
        if item is None:
            continue
        subject_items = by_subject.setdefault(item["subject"], [])
        if len(subject_items) < per_subject:
            subject_items.append(item)

    if len(by_subject) != MMLU_SUBJECT_COUNT:
        raise RuntimeError(
            "expected %d MMLU subjects at pinned revision, found %d"
            % (MMLU_SUBJECT_COUNT, len(by_subject))
        )
    short = sorted(subject for subject, items in by_subject.items() if len(items) < per_subject)
    if short:
        raise RuntimeError(
            "MMLU subjects have fewer than %d valid four-choice items: %s"
            % (per_subject, ", ".join(short))
        )
    return [item for subject in sorted(by_subject) for item in by_subject[subject]]


def load_arc(limit):
    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="test",
        streaming=True,
        revision=ARC_DATASET_REVISION,
    )
    items = []
    for row in dataset:
        choices = (row.get("choices") or {}).get("text") or []
        labels = (row.get("choices") or {}).get("label") or []
        if len(choices) != 4 or len(labels) != 4:
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
    single = single_token_letters(tokenizer)
    if single is not None:
        with torch.no_grad():
            logits = model(input_ids=prompt_ids).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = [float(log_probs[token]) for token in single]
        return int(max(range(len(scores)), key=lambda i: scores[i]))
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


def build_plans(low_layers):
    low = set(range(low_layers))

    def dense(_layer, _attribute):
        return "fp16"

    def uniform_iq2(_layer, _attribute):
        return "iq2"

    def mixed_stq(layer, attribute):
        if attribute == "down_proj":
            return "iq2"
        return "stq" if layer in low else "iq2"

    def mixed_ltc(layer, attribute):
        if attribute == "down_proj":
            return "iq2"
        return "ltc" if layer in low else "iq2"

    def iq2s_all(_layer, _attribute):
        return "iq2_s"

    def iq2xs_all(_layer, _attribute):
        return "iq2_xs"

    def iq3_all(_layer, _attribute):
        return "iq3_xxs"

    def iq3s_all(_layer, _attribute):
        return "iq3_s"

    def iq3_low_iq2_high(layer, attribute):
        return "iq3_xxs" if layer in low else "iq2_xxs"

    def iq2_low_iq3_high(layer, attribute):
        return "iq2_xxs" if layer in low else "iq3_xxs"

    def ltc_iq3(layer, attribute):
        if attribute == "down_proj":
            return "iq3_xxs"
        return "ltc" if layer in low else "iq3_xxs"

    def dense_iq2(_layer, _attribute):
        return "iq2_xxs"

    def dense_iq3(_layer, _attribute):
        return "iq3_xxs"

    def dense_iq3s(_layer, _attribute):
        return "iq3_s"

    def dense_iq3_ref(_layer, _attribute):
        return "iq3_xxs_ref"

    def dense_fp16(_layer, _attribute):
        return "fp16"

    return {
        "dense": dense,
        "uniform_iq2": uniform_iq2,
        "mixed_stq": mixed_stq,
        "mixed_ltc": mixed_ltc,
        "iq2s_all": iq2s_all,
        "iq2xs_all": iq2xs_all,
        "iq3_all": iq3_all,
        "iq3s_all": iq3s_all,
        "iq3_low_iq2_high": iq3_low_iq2_high,
        "iq2_low_iq3_high": iq2_low_iq3_high,
        "ltc_iq3": ltc_iq3,
        "dense_iq2": dense_iq2,
        "dense_iq3": dense_iq3,
        "dense_iq3s": dense_iq3s,
        "dense_iq3_ref": dense_iq3_ref,
        "dense_fp16": dense_fp16,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="paired task accuracy across quantization arms")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--imatrix", required=True)
    mmlu = parser.add_mutually_exclusive_group()
    mmlu.add_argument("--mmlu", type=int)
    mmlu.add_argument("--mmlu-per-subject", type=int)
    parser.add_argument("--arc", type=int, default=230)
    parser.add_argument("--low-layers", type=int, default=6)
    parser.add_argument("--arms", default="dense,mixed_stq,mixed_ltc")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mmlu is None and args.mmlu_per_subject is None:
        args.mmlu = 140
    if args.mmlu is not None and args.mmlu < 0:
        parser.error("--mmlu must be non-negative")
    if args.mmlu_per_subject is not None and args.mmlu_per_subject <= 0:
        parser.error("--mmlu-per-subject must be positive")
    if args.arc < 0:
        parser.error("--arc must be non-negative")
    return args


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    importance = torch.load(Path(args.imatrix).with_suffix(".pt"), map_location="cpu")

    if args.mmlu_per_subject is None:
        mmlu_items = load_mmlu(args.mmlu)
        mmlu_count = args.mmlu
        mmlu_sampling = {"mode": "prefix", "count": args.mmlu}
    else:
        mmlu_items = load_mmlu_stratified(args.mmlu_per_subject)
        mmlu_count = len(mmlu_items)
        mmlu_sampling = {
            "mode": "stratified_per_subject",
            "per_subject": args.mmlu_per_subject,
            "subjects": MMLU_SUBJECT_COUNT,
            "count": mmlu_count,
        }
    arc_items = load_arc(args.arc)
    items = mmlu_items + arc_items
    sampling_scheme = {
        "mmlu": mmlu_sampling,
        "arc": {"mode": "valid_4_choice_prefix", "count": args.arc},
    }
    print("items: %d (mmlu %d, arc %d)" % (len(items), mmlu_count, args.arc), flush=True)
    fingerprint = item_fingerprint(items)

    plans = build_plans(args.low_layers)
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]

    results = {}
    details = {}
    provenance_by_arm = {}
    cache_dir = Path(args.out).parent
    for name in selected:
        if name not in plans:
            raise ValueError("unknown arm " + name)
        provenance = cache_provenance(
            args.model,
            args.revision,
            args.dtype,
            mmlu_count,
            args.arc,
            args.low_layers,
            name,
            sampling_scheme,
            fingerprint,
        )
        provenance_by_arm[name] = provenance
        cached = cache_dir / ("correct_%s.json" % name)
        if cached.is_file():
            correct = load_cached_correct(cached, provenance, len(items))
            if correct is not None:
                results[name] = correct
                details[name] = {"accuracy": sum(correct) / len(correct), "correct": sum(correct)}
                print("[arm] %s reused from %s" % (name, cached.name), flush=True)
                continue
            print("[arm] %s rejected cache %s" % (name, cached.name), flush=True)
        print("[arm] %s" % name, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=getattr(torch, args.dtype),
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
            print("  bpw %.4f mean_error %.4f" % (plan_stats["bpw"], plan_stats["mean_error"]),
                  flush=True)
        correct = run_arm(model, tokenizer, items, device)
        results[name] = correct
        accuracy = sum(correct) / len(correct)
        details[name] = {"accuracy": accuracy, "correct": sum(correct)}
        cached.write_text(
            json.dumps({"provenance": provenance, "correct": correct}), encoding="utf-8"
        )
        print("  accuracy %.4f (%d/%d)" % (accuracy, sum(correct), len(correct)), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = compare(results, "dense")
    report["provenance"] = provenance_by_arm
    report["correct_vectors"] = results
    report["items_detail"] = [
        {"task": i["task"], "subject": i["subject"]} for i in items
    ]
    report["config"] = {
        "model": args.model,
        "revision": args.revision,
        "low_layers": args.low_layers,
        "mmlu": mmlu_count,
        "mmlu_per_subject": args.mmlu_per_subject,
        "arc": args.arc,
        "dtype": args.dtype,
        "sampling_scheme": sampling_scheme,
        "dataset_revisions": {
            "cais/mmlu": MMLU_DATASET_REVISION,
            "allenai/ai2_arc": ARC_DATASET_REVISION,
        },
        "ordered_item_fingerprint": fingerprint,
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
