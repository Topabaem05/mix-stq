from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset
from eval_mixed import apply_plan
from task_accuracy import compare
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]
MMLU_DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
ARC_DATASET_REVISION = "210d026faf9955653af8916fad021475a3f00453"
MMLU_SUBJECT_COUNT = 57
STRICT_PROTOCOL = "qwen38_bf16_800"
STRICT_PROTOCOL_MODEL = "Qwen/Qwen3.8-27B"
STRICT_PROTOCOL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
STRICT_PROTOCOL_FINGERPRINT = "a72515282c6fc20f34188b3102d99468ab2b02266105ed9c6e4ec405fbad8fd0"
STRICT_PROTOCOL_ARMS = ["dense", "dense_iq3_ref"]


def item_fingerprint(items):
    canonical = json.dumps(
        items, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_identity(device, requested_dtype):
    gpu = None
    if device == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "device": device,
        "requested_dtype": requested_dtype,
    }


def parameter_dtype_distribution(model):
    distribution = {}
    for parameter in model.parameters():
        if not parameter.is_floating_point():
            continue
        dtype = str(parameter.dtype)
        distribution[dtype] = distribution.get(dtype, 0) + parameter.numel()
    return dict(sorted(distribution.items()))


def validate_bfloat16_distribution(distribution, arm):
    valid = (
        isinstance(distribution, dict)
        and set(distribution) == {"torch.bfloat16"}
        and type(distribution["torch.bfloat16"]) is int
        and distribution["torch.bfloat16"] > 0
    )
    if not valid:
        raise RuntimeError(
            "strict BF16 arm %s loaded floating parameter elements by dtype %s"
            % (arm, distribution)
        )


def cache_provenance(
    model,
    revision,
    dtype,
    mmlu,
    arc,
    low_layers,
    arm,
    protocol,
    sampling_scheme,
    fingerprint,
    runtime,
):
    return {
        "model": model,
        "revision": revision,
        "dtype": dtype,
        "mmlu": mmlu,
        "arc": arc,
        "low_layers": low_layers,
        "arm": arm,
        "protocol": protocol,
        "sampling_scheme": sampling_scheme,
        "dataset_revisions": {
            "cais/mmlu": MMLU_DATASET_REVISION,
            "allenai/ai2_arc": ARC_DATASET_REVISION,
        },
        "ordered_item_fingerprint": fingerprint,
        "runtime": runtime,
    }


def load_cached_result(path, provenance, item_count):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    correct = payload.get("correct")
    execution = payload.get("execution")
    if payload.get("provenance") != provenance:
        return None
    if not isinstance(correct, list) or len(correct) != item_count:
        return None
    if any(type(value) is not int or value not in (0, 1) for value in correct):
        return None
    required_execution = {
        "requested_dtype",
        "parameter_elements_by_dtype_before_plan",
        "parameter_elements_by_dtype_after_plan",
        "plan_stats",
    }
    if not isinstance(execution, dict) or set(execution) != required_execution:
        return None
    if execution["requested_dtype"] != provenance["dtype"]:
        return None
    if provenance["protocol"] == STRICT_PROTOCOL:
        try:
            validate_bfloat16_distribution(
                execution["parameter_elements_by_dtype_before_plan"], provenance["arm"]
            )
            validate_bfloat16_distribution(
                execution["parameter_elements_by_dtype_after_plan"], provenance["arm"]
            )
        except RuntimeError:
            return None
    return correct, execution


def load_cached_correct(path, provenance, item_count):
    cached = load_cached_result(path, provenance, item_count)
    return None if cached is None else cached[0]


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
    answer = row.get("answer")
    if len(choices) != 4 or type(answer) is not int or answer not in range(4):
        return None
    return {
        "task": "mmlu",
        "subject": row.get("subject", ""),
        "question": (row.get("question") or "").strip(),
        "choices": [str(c).strip() for c in choices],
        "answer": answer,
    }


def load_mmlu(limit, subjects=None):
    if limit == 0:
        return []
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
    if limit == 0:
        return []
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


def validate_item_counts(mmlu_items, arc_items, expected_mmlu, expected_arc):
    if len(mmlu_items) != expected_mmlu:
        raise RuntimeError(
            "expected %d MMLU items, loaded %d" % (expected_mmlu, len(mmlu_items))
        )
    if len(arc_items) != expected_arc:
        raise RuntimeError("expected %d ARC items, loaded %d" % (expected_arc, len(arc_items)))
    expected_total = expected_mmlu + expected_arc
    actual_total = len(mmlu_items) + len(arc_items)
    if actual_total != expected_total:
        raise RuntimeError("expected %d total items, loaded %d" % (expected_total, actual_total))
    if actual_total == 0:
        raise RuntimeError("evaluation requires at least one item")


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


def validate_protocol(args):
    if args.protocol == "generic":
        return
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]
    required = {
        "model": (args.model, STRICT_PROTOCOL_MODEL),
        "revision": (args.revision, STRICT_PROTOCOL_REVISION),
        "dtype": (args.dtype, "bfloat16"),
        "mmlu_per_subject": (args.mmlu_per_subject, 10),
        "arc": (args.arc, 230),
        "arms": (selected, STRICT_PROTOCOL_ARMS),
    }
    mismatches = [
        "%s=%r (required %r)" % (field, actual, expected)
        for field, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("protocol %s requires %s" % (STRICT_PROTOCOL, "; ".join(mismatches)))


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
    parser.add_argument("--protocol", choices=("generic", STRICT_PROTOCOL), default="generic")
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
    validate_protocol(args)

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
    validate_item_counts(mmlu_items, arc_items, mmlu_count, args.arc)
    items = mmlu_items + arc_items
    sampling_scheme = {
        "mmlu": mmlu_sampling,
        "arc": {"mode": "valid_4_choice_prefix", "count": args.arc},
    }
    fingerprint = item_fingerprint(items)
    if args.protocol == STRICT_PROTOCOL and fingerprint != STRICT_PROTOCOL_FINGERPRINT:
        raise RuntimeError(
            "protocol %s expected ordered item fingerprint %s, got %s"
            % (STRICT_PROTOCOL, STRICT_PROTOCOL_FINGERPRINT, fingerprint)
        )
    print("items: %d (mmlu %d, arc %d)" % (len(items), mmlu_count, args.arc), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = runtime_identity(device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    importance = torch.load(Path(args.imatrix).with_suffix(".pt"), map_location="cpu")

    plans = build_plans(args.low_layers)
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]

    results = {}
    details = {}
    provenance_by_arm = {}
    execution_by_arm = {}
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
            args.protocol,
            sampling_scheme,
            fingerprint,
            runtime,
        )
        provenance_by_arm[name] = provenance
        cached = cache_dir / ("correct_%s.json" % name)
        if cached.is_file():
            cached_result = load_cached_result(cached, provenance, len(items))
            if cached_result is not None:
                correct, execution = cached_result
                results[name] = correct
                execution_by_arm[name] = execution
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
        before_plan = parameter_dtype_distribution(model)
        if args.protocol == STRICT_PROTOCOL:
            validate_bfloat16_distribution(before_plan, name)
        plan_stats = None
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
        execution = {
            "requested_dtype": args.dtype,
            "parameter_elements_by_dtype_before_plan": before_plan,
            "parameter_elements_by_dtype_after_plan": parameter_dtype_distribution(model),
            "plan_stats": plan_stats,
        }
        correct = run_arm(model, tokenizer, items, device)
        results[name] = correct
        execution_by_arm[name] = execution
        accuracy = sum(correct) / len(correct)
        details[name] = {"accuracy": accuracy, "correct": sum(correct)}
        cached.write_text(
            json.dumps({
                "provenance": provenance,
                "execution": execution,
                "correct": correct,
            }),
            encoding="utf-8",
        )
        print("  accuracy %.4f (%d/%d)" % (accuracy, sum(correct), len(correct)), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = compare(results, "dense")
    report["runtime"] = runtime
    report["provenance"] = provenance_by_arm
    report["execution"] = execution_by_arm
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
        "protocol": args.protocol,
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
