from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import uuid
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
STRICT_CACHE_SCHEMA = 2
STRICT_IMATRIX_SIZE = 7137641
STRICT_IMATRIX_SHA256 = "def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0"
STRICT_PLAN_PARAMS = 17112760320
STRICT_PLAN_BYTES = 6550978560
STRICT_PLAN_BPW = 3.0625
SOURCE_IDENTITY_FILES = (
    "eval_tasks.py",
    "eval_mixed.py",
    "iq3_vectorized.py",
    "torch_iq2.py",
    "tier_tables.json",
    "task_accuracy.py",
)


def item_fingerprint(items):
    canonical = json.dumps(
        items, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    source_dir = Path(__file__).resolve().parent
    return {
        filename: sha256_file(source_dir / filename)
        for filename in SOURCE_IDENTITY_FILES
    }


def resolve_imatrix_identity(path, strict=False):
    resolved = Path(path)
    if resolved.suffix != ".pt":
        resolved = resolved.with_suffix(".pt")
    resolved = resolved.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("importance tensor not found: %s" % resolved)
    identity = {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if strict and (
        identity["size"] != STRICT_IMATRIX_SIZE
        or identity["sha256"] != STRICT_IMATRIX_SHA256
    ):
        raise RuntimeError(
            "strict imatrix identity mismatch for %s: size %d sha256 %s"
            % (resolved, identity["size"], identity["sha256"])
        )
    return identity


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=1, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(temporary_name).unlink()
        raise


def artifact_paths(output_path):
    output = Path(output_path)
    return {
        "result": output,
        "completion": output.with_name(output.name + ".complete.json"),
        "progress": output.with_name(output.name + ".progress.json"),
        "failure": output.with_name(output.name + ".failure.json"),
    }


def gpu_preflight(run_command=subprocess.run):
    if not torch.cuda.is_available():
        raise RuntimeError("strict protocol requires CUDA")

    def query(command):
        try:
            completed = run_command(
                command, check=False, capture_output=True, text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("nvidia-smi command failed: %s" % exc) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "nvidia-smi command failed with status %d: %s"
                % (completed.returncode, completed.stderr.strip())
            )
        return completed.stdout.strip()

    gpu_output = query([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    gpu_lines = [line.strip() for line in gpu_output.splitlines() if line.strip()]
    if len(gpu_lines) != 1:
        raise RuntimeError("strict protocol requires exactly one parseable GPU")
    fields = [field.strip() for field in gpu_lines[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("could not parse nvidia-smi GPU inventory")
    try:
        gpu_index = int(fields[0])
        total_memory = int(fields[3])
        free_memory = int(fields[4])
    except ValueError as exc:
        raise RuntimeError("could not parse nvidia-smi GPU memory") from exc
    if total_memory < 92160:
        raise RuntimeError(
            "strict protocol requires at least 92160 MiB GPU memory, got %d" % total_memory
        )

    process_output = query([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    process_lines = [line.strip() for line in process_output.splitlines() if line.strip()]
    if process_lines:
        raise RuntimeError("strict protocol requires an idle GPU; compute processes: %s" % (
            "; ".join(process_lines)
        ))
    return {
        "gpu": {
            "index": gpu_index,
            "name": fields[1],
            "driver": fields[2],
            "total_memory_mib": total_memory,
            "free_memory_mib": free_memory,
        },
        "compute_processes": [],
    }


def validate_strict_plan_stats(arm, plan_stats):
    if arm == "dense":
        if plan_stats is not None:
            raise RuntimeError("strict dense requires null plan_stats")
        return
    if arm != "dense_iq3_ref" or not isinstance(plan_stats, dict):
        raise RuntimeError("strict arm %s requires complete plan_stats" % arm)
    inventory = plan_stats.get("tensor_inventory")
    skipped = plan_stats.get("skipped_targets")
    if not isinstance(inventory, list) or len(inventory) != 192:
        raise RuntimeError("strict dense_iq3_ref requires 192 tensor records")
    if skipped != []:
        raise RuntimeError("strict dense_iq3_ref requires no skipped targets")
    identities = []
    layer_attributes = []
    parameter_total = 0
    byte_total = 0
    required_fields = {
        "module_name", "weight_name", "layer", "attribute", "tier", "params", "bpw"
    }
    for record in inventory:
        if not isinstance(record, dict) or not required_fields.issubset(record):
            raise RuntimeError("strict tensor inventory record is incomplete")
        if (
            not isinstance(record["module_name"], str)
            or not isinstance(record["weight_name"], str)
            or record["weight_name"] != record["module_name"] + ".weight"
            or type(record["layer"]) is not int
            or record["attribute"] not in ("gate_proj", "up_proj", "down_proj")
            or not record["module_name"].endswith(
                ".layers.%d.mlp.%s" % (record["layer"], record["attribute"])
            )
            or record["tier"] != "iq3_xxs_ref"
            or type(record["params"]) is not int
            or record["params"] <= 0
            or not math.isclose(float(record["bpw"]), STRICT_PLAN_BPW)
        ):
            raise RuntimeError("strict tensor inventory record has invalid identity or tier")
        identities.append(record["weight_name"])
        layer_attributes.append((record["layer"], record["attribute"]))
        parameter_total += record["params"]
        if "bytes" in record:
            if type(record["bytes"]) is not int or record["bytes"] < 0:
                raise RuntimeError("strict tensor inventory record has invalid bytes")
            byte_total += record["bytes"]
    expected_layer_attributes = {
        (layer, attribute)
        for layer in range(64)
        for attribute in ("gate_proj", "up_proj", "down_proj")
    }
    if len(set(identities)) != 192:
        raise RuntimeError("strict dense_iq3_ref requires 192 unique tensor identities")
    if identities != sorted(identities):
        raise RuntimeError("strict dense_iq3_ref tensor inventory is not deterministic")
    if set(layer_attributes) != expected_layer_attributes or len(layer_attributes) != len(
            set(layer_attributes)):
        raise RuntimeError("strict dense_iq3_ref layer/attribute inventory is incomplete")
    if plan_stats.get("params") != STRICT_PLAN_PARAMS or parameter_total != STRICT_PLAN_PARAMS:
        raise RuntimeError("strict dense_iq3_ref parameter total mismatch")
    if plan_stats.get("bytes") != STRICT_PLAN_BYTES:
        raise RuntimeError("strict dense_iq3_ref byte total mismatch")
    if all("bytes" in record for record in inventory) and byte_total != STRICT_PLAN_BYTES:
        raise RuntimeError("strict dense_iq3_ref inventory byte total mismatch")
    try:
        bpw = float(plan_stats.get("bpw"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("strict dense_iq3_ref bpw is invalid") from exc
    if not math.isclose(bpw, STRICT_PLAN_BPW):
        raise RuntimeError("strict dense_iq3_ref bpw mismatch")


def validate_strict_execution(arm, execution):
    required = {
        "requested_dtype",
        "parameter_elements_by_dtype_before_plan",
        "parameter_elements_by_dtype_after_plan",
        "plan_stats",
    }
    if not isinstance(execution, dict) or set(execution) != required:
        raise RuntimeError("strict arm %s execution evidence is incomplete" % arm)
    if execution["requested_dtype"] != "bfloat16":
        raise RuntimeError("strict arm %s requested dtype is not bfloat16" % arm)
    validate_bfloat16_distribution(execution["parameter_elements_by_dtype_before_plan"], arm)
    validate_bfloat16_distribution(execution["parameter_elements_by_dtype_after_plan"], arm)
    validate_strict_plan_stats(arm, execution["plan_stats"])


def build_strict_decision(comparison):
    try:
        delta = float(comparison["accuracy_delta"])
        low, high = (float(value) for value in comparison["ci_95"])
        p_value = float(comparison["mcnemar_p"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("strict comparison statistics are incomplete") from exc
    noninferior = high <= 0.02
    significant_dense_advantage = low > 0.0 and p_value < 0.05
    iq3_advantage_signal = high < 0.0 and p_value < 0.05
    if iq3_advantage_signal:
        primary = "iq3_advantage_signal"
    elif noninferior:
        primary = "noninferior"
    elif significant_dense_advantage:
        primary = "significant_dense_advantage"
    else:
        primary = "inconclusive"
    return {
        "comparison": "dense_vs_dense_iq3_ref",
        "margin": 0.02,
        "raw": {
            "accuracy_delta": delta,
            "ci_95": [low, high],
            "mcnemar_p": p_value,
        },
        "noninferior": noninferior,
        "significant_dense_advantage": significant_dense_advantage,
        "iq3_advantage_signal": iq3_advantage_signal,
        "primary": primary,
    }


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
    imatrix_identity=None,
    source_hashes=None,
):
    provenance = {
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
    if protocol == STRICT_PROTOCOL:
        provenance.update({
            "cache_schema": STRICT_CACHE_SCHEMA,
            "imatrix": imatrix_identity,
            "source_sha256": source_hashes,
        })
    return provenance


def load_cached_result(path, provenance, item_count):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if provenance.get("protocol") == STRICT_PROTOCOL and (
        payload.get("status") != "complete"
        or not isinstance(payload.get("run_id"), str)
        or not payload["run_id"].strip()
    ):
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
        if (
            provenance.get("cache_schema") != STRICT_CACHE_SCHEMA
            or not isinstance(provenance.get("imatrix"), dict)
            or not isinstance(provenance.get("source_sha256"), dict)
        ):
            return None
        try:
            validate_strict_execution(provenance["arm"], execution)
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


def run_arm(model, tokenizer, items, device, on_progress=None):
    correct = []
    for item in items:
        prediction = score_item(model, tokenizer, item, device)
        correct.append(int(prediction == item["answer"]))
        if on_progress is not None:
            on_progress(len(correct))
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


def quarantine_cache(path, run_id):
    quarantine = path.with_name(path.name + ".quarantine-" + run_id)
    os.replace(path, quarantine)
    return quarantine


def run_evaluation(args, run_id, paths, progress):
    strict = args.protocol == STRICT_PROTOCOL
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
    if strict and fingerprint != STRICT_PROTOCOL_FINGERPRINT:
        raise RuntimeError(
            "protocol %s expected ordered item fingerprint %s, got %s"
            % (STRICT_PROTOCOL, STRICT_PROTOCOL_FINGERPRINT, fingerprint)
        )
    print("items: %d (mmlu %d, arc %d)" % (len(items), mmlu_count, args.arc), flush=True)

    imatrix = resolve_imatrix_identity(args.imatrix, strict=strict)
    source_hashes = source_identity() if strict else None
    preflight = gpu_preflight() if strict else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = runtime_identity(device, args.dtype)
    if strict:
        runtime["gpu_preflight"] = preflight
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    importance = torch.load(imatrix["path"], map_location="cpu")

    plans = build_plans(args.low_layers)
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]
    results = {}
    provenance_by_arm = {}
    execution_by_arm = {}
    cache_dir = paths["result"].parent
    quarantined_caches = []
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
            imatrix_identity=imatrix if strict else None,
            source_hashes=source_hashes,
        )
        provenance_by_arm[name] = provenance
        cached = cache_dir / ("correct_%s.json" % name)
        if cached.is_file():
            cached_result = load_cached_result(cached, provenance, len(items))
            if cached_result is not None:
                correct, execution = cached_result
                results[name] = correct
                execution_by_arm[name] = execution
                progress["completed_items"][name] = len(correct)
                progress["completed_arms"].append(name)
                atomic_write_json(paths["progress"], progress)
                print("[arm] %s reused from %s" % (name, cached.name), flush=True)
                continue
            if strict:
                quarantined = quarantine_cache(cached, run_id)
                quarantined_caches.append(str(quarantined))
                print("[arm] %s quarantined cache %s" % (name, quarantined.name), flush=True)
            else:
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
        if strict:
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
            print("  bpw %.4f mean_error %.4f" % (
                plan_stats["bpw"], plan_stats["mean_error"]), flush=True)
        execution = {
            "requested_dtype": args.dtype,
            "parameter_elements_by_dtype_before_plan": before_plan,
            "parameter_elements_by_dtype_after_plan": parameter_dtype_distribution(model),
            "plan_stats": plan_stats,
        }
        if strict:
            validate_strict_execution(name, execution)

        def record_item_count(count, arm=name):
            progress["completed_items"][arm] = count
            atomic_write_json(paths["progress"], progress)

        correct = run_arm(model, tokenizer, items, device, record_item_count)
        results[name] = correct
        execution_by_arm[name] = execution
        progress["completed_items"][name] = len(correct)
        progress["completed_arms"].append(name)
        atomic_write_json(paths["progress"], progress)
        atomic_write_json(cached, {
            "run_id": run_id,
            "status": "complete",
            "provenance": provenance,
            "execution": execution,
            "correct": correct,
        })
        accuracy = sum(correct) / len(correct)
        print("  accuracy %.4f (%d/%d)" % (accuracy, sum(correct), len(correct)), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = compare(results, "dense")
    report["run_id"] = run_id
    report["status"] = "complete"
    report["runtime"] = runtime
    report["provenance"] = provenance_by_arm
    report["execution"] = execution_by_arm
    report["correct_vectors"] = results
    report["items_detail"] = [
        {"task": item["task"], "subject": item["subject"]} for item in items
    ]
    report["artifact_identity"] = {
        "imatrix": imatrix,
        "source_sha256": source_hashes,
        "quarantined_caches": quarantined_caches,
    }
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
    if strict:
        comparison = report["comparisons"].get("dense_vs_dense_iq3_ref")
        if comparison is None:
            raise RuntimeError("strict result is missing dense_vs_dense_iq3_ref")
        report["decision"] = build_strict_decision(comparison)
    atomic_write_json(paths["result"], report)
    result_sha256 = sha256_file(paths["result"])
    atomic_write_json(paths["completion"], {
        "run_id": run_id,
        "status": "complete",
        "result": str(paths["result"].resolve()),
        "result_sha256": result_sha256,
    })
    progress["status"] = "complete"
    progress["result_sha256"] = result_sha256
    atomic_write_json(paths["progress"], progress)

    print()
    for name, stats in report["arms"].items():
        print("%-14s %.4f (%d/%d)" % (
            name, stats["accuracy"], stats["correct"], report["items"]))
    print()
    for label, comparison in report["comparisons"].items():
        print("%-28s delta=%+.4f CI[%+.4f, %+.4f] p=%.4f %s" % (
            label, comparison["accuracy_delta"], comparison["ci_95"][0],
            comparison["ci_95"][1], comparison["mcnemar_p"],
            "SIGNIFICANT" if comparison["significant"] else "not significant"))
    print()
    print("wrote %s" % args.out)


def main() -> int:
    args = parse_args()
    paths = artifact_paths(args.out)
    if args.protocol == STRICT_PROTOCOL:
        existing = [paths[name] for name in ("result", "completion") if paths[name].exists()]
        if existing:
            raise FileExistsError(
                "strict run refuses to overwrite completed artifacts: %s"
                % ", ".join(str(path) for path in existing)
            )
    run_id = uuid.uuid4().hex
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]
    progress = {
        "run_id": run_id,
        "status": "running",
        "protocol": args.protocol,
        "completed_arms": [],
        "completed_items": {name: 0 for name in selected},
    }
    atomic_write_json(paths["progress"], progress)
    try:
        validate_protocol(args)
        run_evaluation(args, run_id, paths, progress)
    except BaseException as exc:
        atomic_write_json(paths["failure"], {
            "run_id": run_id,
            "status": "failed",
            "protocol": args.protocol,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_arms": list(progress["completed_arms"]),
            "completed_items": dict(progress["completed_items"]),
        })
        progress["status"] = "failed"
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        atomic_write_json(paths["progress"], progress)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
