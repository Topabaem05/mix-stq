from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import torch

stub_datasets = types.ModuleType("datasets")
stub_datasets.load_dataset = lambda *a, **k: iter([])
stub_datasets.__version__ = "test-datasets"
sys.modules.setdefault("datasets", stub_datasets)

stub_tf = types.ModuleType("transformers")
stub_tf.AutoModelForCausalLM = object
stub_tf.AutoTokenizer = object
stub_tf.__version__ = "test-transformers"
sys.modules.setdefault("transformers", stub_tf)

stub_iq2 = types.ModuleType("torch_iq2")
stub_iq2.quantize_rows = lambda *a, **k: (None, 0.0)
sys.modules.setdefault("torch_iq2", stub_iq2)
stub_ltc = types.ModuleType("torch_ltc")
stub_ltc.quantize_rows = lambda *a, **k: (None, 0.0)
sys.modules.setdefault("torch_ltc", stub_ltc)

import eval_tasks  # noqa: E402
from eval_tasks import (  # noqa: E402
    ARC_DATASET_REVISION,
    LETTERS,
    MMLU_DATASET_REVISION,
    STRICT_PROTOCOL_FINGERPRINT,
    cache_provenance,
    item_fingerprint,
    load_arc,
    load_cached_correct,
    load_cached_result,
    load_mmlu,
    load_mmlu_stratified,
    mmlu_item,
    parameter_dtype_distribution,
    parse_args,
    render,
    runtime_identity,
    score_item,
    validate_bfloat16_distribution,
    validate_item_counts,
    validate_protocol,
)

failures = []

required_new_api = {
    name: getattr(eval_tasks, name, None)
    for name in (
        "STRICT_CACHE_SCHEMA",
        "STRICT_IMATRIX_SHA256",
        "STRICT_IMATRIX_SIZE",
        "atomic_write_json",
        "build_strict_decision",
        "gpu_preflight",
        "resolve_imatrix_identity",
        "source_identity",
        "validate_strict_execution",
        "validate_strict_plan_stats",
    )
}
missing_new_api = sorted(name for name, value in required_new_api.items() if value is None)
if missing_new_api:
    failures.append("final strict gate API is missing: %s" % ", ".join(missing_new_api))


def complete_strict_plan_stats():
    per_tensor_params = 89128960
    inventory = []
    for layer in range(64):
        for attribute in ("gate_proj", "up_proj", "down_proj"):
            module_name = "model.language_model.layers.%d.mlp.%s" % (layer, attribute)
            inventory.append({
                "module_name": module_name,
                "weight_name": module_name + ".weight",
                "layer": layer,
                "attribute": attribute,
                "tier": "iq3_xxs_ref",
                "params": per_tensor_params,
                "bpw": 3.0625,
            })
    inventory.sort(key=lambda record: record["weight_name"])
    return {
        "params": 17112760320,
        "bytes": 6550978560,
        "bpw": 3.0625,
        "mean_error": 0.0,
        "tensor_inventory": inventory,
        "skipped_targets": [],
    }


if not missing_new_api:
    atomic_write_json = required_new_api["atomic_write_json"]
    build_strict_decision = required_new_api["build_strict_decision"]
    gpu_preflight = required_new_api["gpu_preflight"]
    resolve_imatrix_identity = required_new_api["resolve_imatrix_identity"]
    source_identity = required_new_api["source_identity"]
    validate_strict_execution = required_new_api["validate_strict_execution"]
    validate_strict_plan_stats = required_new_api["validate_strict_plan_stats"]

    valid_plan_stats = complete_strict_plan_stats()
    validate_strict_plan_stats("dense", None)
    validate_strict_plan_stats("dense_iq3_ref", valid_plan_stats)
    invalid_plan_stats = []
    for mutation in ("null", "partial", "tier", "layer", "duplicate", "skipped", "params"):
        candidate = None if mutation == "null" else json.loads(json.dumps(valid_plan_stats))
        if mutation == "partial":
            candidate["tensor_inventory"].pop()
        elif mutation == "tier":
            candidate["tensor_inventory"][0]["tier"] = "iq3_xxs"
        elif mutation == "layer":
            candidate["tensor_inventory"][0]["layer"] = 64
        elif mutation == "duplicate":
            candidate["tensor_inventory"][0] = dict(candidate["tensor_inventory"][1])
        elif mutation == "skipped":
            candidate["skipped_targets"] = [{"reason": "missing_importance"}]
        elif mutation == "params":
            candidate["params"] -= 1
        invalid_plan_stats.append((mutation, candidate))
    for mutation, candidate in invalid_plan_stats:
        try:
            validate_strict_plan_stats("dense_iq3_ref", candidate)
        except RuntimeError:
            pass
        else:
            failures.append("strict plan validation accepted %s evidence" % mutation)
    try:
        validate_strict_plan_stats("dense", valid_plan_stats)
    except RuntimeError:
        pass
    else:
        failures.append("strict dense accepted non-null plan evidence")

    valid_strict_arm_execution = {
        "requested_dtype": "bfloat16",
        "parameter_elements_by_dtype_before_plan": {"torch.bfloat16": 2},
        "parameter_elements_by_dtype_after_plan": {"torch.bfloat16": 2},
        "plan_stats": valid_plan_stats,
    }
    validate_strict_execution("dense_iq3_ref", valid_strict_arm_execution)
    try:
        validate_strict_execution(
            "dense_iq3_ref",
            dict(valid_strict_arm_execution,
                 parameter_elements_by_dtype_after_plan={"torch.float32": 2}),
        )
    except RuntimeError:
        pass
    else:
        failures.append("fresh strict execution accepted a mixed post-plan dtype")

    decision_cases = [
        ((-0.03, -0.01, 0.01), "iq3_advantage_signal"),
        ((0.001, 0.015, 0.01), "noninferior"),
        ((0.021, 0.03, 0.01), "significant_dense_advantage"),
        ((-0.01, 0.03, 0.50), "inconclusive"),
    ]
    for (low, high, p_value), expected_branch in decision_cases:
        comparison = {
            "accuracy_delta": (low + high) / 2,
            "ci_95": [low, high],
            "mcnemar_p": p_value,
        }
        decision = build_strict_decision(comparison)
        if decision.get("primary") != expected_branch:
            failures.append("strict decision selected %r instead of %r" % (
                decision.get("primary"), expected_branch))
        if decision.get("margin") != 0.02:
            failures.append("strict decision changed the frozen noninferiority margin")
        if decision.get("raw") != comparison:
            failures.append("strict decision did not preserve raw comparison statistics")
    overlapping = build_strict_decision({
        "accuracy_delta": 0.008,
        "ci_95": [0.001, 0.015],
        "mcnemar_p": 0.01,
    })
    if not overlapping.get("noninferior") or not overlapping.get(
            "significant_dense_advantage") or overlapping.get("primary") != "noninferior":
        failures.append("strict decision priority mishandled a small significant dense advantage")

    class Completed:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def successful_smi(command, **_kwargs):
        if any("query-compute-apps" in part for part in command):
            return Completed("")
        return Completed("0, NVIDIA H100 NVL, 570.86.15, 97871, 96800\n")

    original_cuda_available = eval_tasks.torch.cuda.is_available
    eval_tasks.torch.cuda.is_available = lambda: True
    try:
        preflight = gpu_preflight(run_command=successful_smi)
        if preflight.get("gpu", {}).get("total_memory_mib") != 97871:
            failures.append("GPU preflight did not parse total memory")

        def occupied_smi(command, **kwargs):
            if any("query-compute-apps" in part for part in command):
                return Completed("GPU-uuid, 123, python, 1000\n")
            return successful_smi(command, **kwargs)

        try:
            gpu_preflight(run_command=occupied_smi)
        except RuntimeError:
            pass
        else:
            failures.append("GPU preflight accepted an existing compute process")
    finally:
        eval_tasks.torch.cuda.is_available = original_cuda_available

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "atomic.json"
        atomic_write_json(target, {"status": "complete", "value": 1})
        if json.loads(target.read_text(encoding="utf-8")) != {
                "status": "complete", "value": 1}:
            failures.append("atomic JSON writer did not persist the complete payload")
        if list(Path(tmp).glob(".*.tmp")):
            failures.append("atomic JSON writer left a temporary file after success")

        imatrix_path = Path(tmp) / "importance.pt"
        imatrix_path.write_bytes(b"test-imatrix")
        original_size = eval_tasks.STRICT_IMATRIX_SIZE
        original_sha = eval_tasks.STRICT_IMATRIX_SHA256
        eval_tasks.STRICT_IMATRIX_SIZE = imatrix_path.stat().st_size
        eval_tasks.STRICT_IMATRIX_SHA256 = hashlib.sha256(b"test-imatrix").hexdigest()
        try:
            identity = resolve_imatrix_identity(imatrix_path, strict=True)
            if identity != {
                "path": str(imatrix_path.resolve()),
                "size": len(b"test-imatrix"),
                "sha256": hashlib.sha256(b"test-imatrix").hexdigest(),
            }:
                failures.append("strict imatrix identity did not bind path, size, and SHA-256")
            imatrix_path.write_bytes(b"wrong")
            try:
                resolve_imatrix_identity(imatrix_path, strict=True)
            except RuntimeError:
                pass
            else:
                failures.append("strict imatrix identity accepted wrong bytes")
        finally:
            eval_tasks.STRICT_IMATRIX_SIZE = original_size
            eval_tasks.STRICT_IMATRIX_SHA256 = original_sha

    source_hashes = source_identity()
    if set(source_hashes) != {
        "eval_tasks.py", "eval_mixed.py", "iq3_vectorized.py", "torch_iq2.py",
        "tier_tables.json", "task_accuracy.py",
    } or any(len(digest) != 64 for digest in source_hashes.values()):
        failures.append("source identity did not bind all evaluator and quantizer sources")

item = {
    "task": "mmlu",
    "subject": "test",
    "question": "What is 2+2?",
    "choices": ["3", "4", "5", "6"],
    "answer": 1,
}
text = render(item)
for letter in LETTERS:
    if ("%s. " % letter) not in text:
        failures.append("rendered prompt missing option %s" % letter)
if not text.rstrip().endswith("Answer:"):
    failures.append("prompt must end with the answer cue")
if item["question"] not in text:
    failures.append("prompt missing the question")


class StubTokenizer:
    def __init__(self, favored):
        self.favored = favored

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = [1, 2, 3] if return_tensors else [10 + LETTERS.index(text.strip())]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


class StubModel:
    def __init__(self, favored_token):
        self.favored_token = favored_token

    def __call__(self, input_ids=None):
        length = input_ids.shape[1]
        logits = torch.full((1, length, 32), -5.0)
        logits[0, :, self.favored_token] = 5.0
        return types.SimpleNamespace(logits=logits)


for expected_index, letter in enumerate(LETTERS):
    tokenizer = StubTokenizer(letter)
    model = StubModel(10 + expected_index)
    picked = score_item(model, tokenizer, item, "cpu")
    if picked != expected_index:
        failures.append("scoring picked %d when option %s was favored" % (picked, letter))

for invalid_answer in (True, False, -1, 4, "1", None):
    invalid_row = dict(item, answer=invalid_answer)
    if mmlu_item(invalid_row) is not None:
        failures.append("invalid MMLU answer was accepted: %r" % invalid_answer)
for valid_answer in range(4):
    valid_row = dict(item, answer=valid_answer)
    if mmlu_item(valid_row) is None:
        failures.append("valid MMLU answer was rejected: %d" % valid_answer)

mmlu_rows = []
for subject_index in range(57):
    subject = "subject_%02d" % subject_index
    subject_items = 112 if subject_index < 5 else 10
    for question_index in range(subject_items):
        mmlu_rows.append({
            "subject": subject,
            "question": "%s question %03d" % (subject, question_index),
            "choices": ["a", "b", "c", "d"],
            "answer": question_index % 4,
        })
arc_rows = [{
    "question": "invalid three-choice item",
    "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]},
    "answerKey": "A",
}]
for question_index in range(230):
    arc_rows.append({
        "question": "arc question %03d" % question_index,
        "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A",
    })

dataset_calls = []


def stub_load_dataset(name, config, **kwargs):
    dataset_calls.append((name, config, kwargs))
    return iter(mmlu_rows if name == "cais/mmlu" else arc_rows)


original_load_dataset = eval_tasks.load_dataset
eval_tasks.load_dataset = stub_load_dataset
try:
    legacy_items = load_mmlu(3)
    if [entry["question"] for entry in legacy_items] != [
        "subject_00 question 000",
        "subject_00 question 001",
        "subject_00 question 002",
    ]:
        failures.append("legacy --mmlu prefix sampling changed")

    stratified_items = load_mmlu_stratified(10)
    subject_counts = {}
    for entry in stratified_items:
        subject_counts[entry["subject"]] = subject_counts.get(entry["subject"], 0) + 1
    if len(stratified_items) != 570 or sorted(subject_counts.values()) != [10] * 57:
        failures.append("stratified MMLU did not select 10 valid items from each of 57 subjects")
    if stratified_items != load_mmlu_stratified(10):
        failures.append("stratified MMLU selection was not deterministic")

    arc_items = load_arc(230)
    if len(arc_items) != 230:
        failures.append("ARC did not select 230 valid four-choice items")
finally:
    eval_tasks.load_dataset = original_load_dataset

zero_load_calls = []
eval_tasks.load_dataset = lambda *args, **kwargs: zero_load_calls.append((args, kwargs))
try:
    if load_mmlu(0) != [] or load_arc(0) != []:
        failures.append("zero item limits did not return empty lists")
    if zero_load_calls:
        failures.append("zero item limits unnecessarily opened datasets")
finally:
    eval_tasks.load_dataset = original_load_dataset

for name, _config, kwargs in dataset_calls:
    expected_revision = MMLU_DATASET_REVISION if name == "cais/mmlu" else ARC_DATASET_REVISION
    if kwargs.get("revision") != expected_revision:
        failures.append("%s dataset revision was not pinned" % name)

required_cli = [
    "--model", "org/model",
    "--revision", "revision-a",
    "--imatrix", "imatrix.json",
    "--out", "report.json",
]
defaults = parse_args(required_cli)
if defaults.mmlu != 140 or defaults.mmlu_per_subject is not None:
    failures.append("legacy default --mmlu behavior changed")
if defaults.protocol != "generic":
    failures.append("generic protocol is not the default")
if defaults.arc != 230:
    failures.append("default ARC valid-item count is not 230")
with contextlib.redirect_stderr(io.StringIO()):
    try:
        parse_args([*required_cli, "--mmlu", "140", "--mmlu-per-subject", "10"])
    except SystemExit as exc:
        if exc.code != 2:
            failures.append("ambiguous MMLU CLI exited with an unexpected status")
    else:
        failures.append("ambiguous --mmlu and --mmlu-per-subject CLI was accepted")

strict_cli = [
    *required_cli,
    "--protocol", "qwen38_bf16_800",
    "--model", "Qwen/Qwen3.8-27B",
    "--revision", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "--dtype", "bfloat16",
    "--mmlu-per-subject", "10",
    "--arc", "230",
    "--arms", "dense,dense_iq3_ref",
]
strict_args = parse_args(strict_cli)
try:
    validate_protocol(strict_args)
except ValueError as exc:
    failures.append("valid strict protocol tuple was rejected: %s" % exc)
for field, value in {
    "model": "other/model",
    "revision": "other-revision",
    "dtype": "float16",
    "mmlu_per_subject": 9,
    "arc": 229,
    "arms": "dense",
}.items():
    invalid_args = types.SimpleNamespace(**vars(strict_args))
    setattr(invalid_args, field, value)
    try:
        validate_protocol(invalid_args)
    except ValueError:
        pass
    else:
        failures.append("strict protocol accepted invalid %s" % field)

runtime = runtime_identity("cpu", "bfloat16")
required_runtime_fields = {
    "platform", "python", "gpu", "torch", "cuda", "transformers", "datasets",
    "device", "requested_dtype",
}
if set(runtime) != required_runtime_fields:
    failures.append("runtime identity fields are incomplete")


class DtypeStubModel:
    def parameters(self):
        return iter([
            torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16)),
            torch.nn.Parameter(torch.zeros(2, dtype=torch.float32)),
        ])


distribution = parameter_dtype_distribution(DtypeStubModel())
if distribution != {"torch.bfloat16": 3, "torch.float32": 2}:
    failures.append("parameter dtype distribution did not count parameter elements")
validate_bfloat16_distribution({"torch.bfloat16": 3}, "dense")
for invalid_distribution in ({}, {"torch.float32": 3}, {"torch.bfloat16": 3, "torch.float32": 1}):
    try:
        validate_bfloat16_distribution(invalid_distribution, "dense")
    except RuntimeError:
        pass
    else:
        failures.append("strict BF16 validation accepted %r" % invalid_distribution)

arc_count_item = dict(item, task="arc_challenge", subject="arc")
validate_item_counts([item], [arc_count_item], 1, 1)
try:
    validate_item_counts([], [], 0, 0)
except RuntimeError:
    pass
else:
    failures.append("zero-total evaluation passed item count validation")
for mmlu_items, arc_items, expected_message in (
    ([], [arc_count_item], "expected 1 MMLU items"),
    ([item], [], "expected 1 ARC items"),
):
    try:
        validate_item_counts(mmlu_items, arc_items, 1, 1)
    except RuntimeError as exc:
        if expected_message not in str(exc):
            failures.append("item count validation raised an unclear error: %s" % exc)
    else:
        failures.append("partial item selection passed exact count validation")

fingerprint = item_fingerprint([item])
reordered_item = {key: item[key] for key in reversed(item)}
if item_fingerprint([reordered_item]) != fingerprint:
    failures.append("item fingerprint must be deterministic across dictionary key order")
changed_item = dict(item, question="What is 3+3?")
if item_fingerprint([changed_item]) == fingerprint:
    failures.append("item fingerprint did not change with item content")

sampling_scheme = {
    "mmlu": {"mode": "prefix", "count": 1},
    "arc": {"mode": "valid_4_choice_prefix", "count": 0},
}
provenance = cache_provenance(
    "org/model",
    "revision-a",
    "float16",
    1,
    0,
    6,
    "dense",
    "generic",
    sampling_scheme,
    fingerprint,
    runtime_identity("cpu", "float16"),
)
if provenance.get("dataset_revisions") != {
    "cais/mmlu": MMLU_DATASET_REVISION,
    "allenai/ai2_arc": ARC_DATASET_REVISION,
}:
    failures.append("cache provenance is missing pinned dataset revisions")
if provenance.get("sampling_scheme") != sampling_scheme:
    failures.append("cache provenance is missing the sampling scheme")
if provenance.get("ordered_item_fingerprint") != fingerprint:
    failures.append("cache provenance is missing the ordered item fingerprint")
strict_provenance = cache_provenance(
    "org/model",
    "revision-a",
    "float16",
    1,
    0,
    6,
    "dense",
    "qwen38_bf16_800",
    sampling_scheme,
    fingerprint,
    runtime_identity("cpu", "float16"),
    imatrix_identity={
        "path": "/immutable/qwen38_imatrix.pt",
        "size": 7137641,
        "sha256": "def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0",
    },
    source_hashes=source_hashes,
)
if provenance.get("protocol") != "generic":
    failures.append("generic cache provenance is missing its protocol identity")
if strict_provenance.get("protocol") != "qwen38_bf16_800":
    failures.append("strict cache provenance is missing its protocol identity")
if strict_provenance == provenance:
    failures.append("generic and strict protocol provenance are identical")
execution_evidence = {
    "requested_dtype": "float16",
    "parameter_elements_by_dtype_before_plan": {"torch.float16": 2},
    "parameter_elements_by_dtype_after_plan": {"torch.float16": 2},
    "plan_stats": None,
}
with tempfile.TemporaryDirectory() as tmp:
    cache = Path(tmp) / "correct_dense.json"
    cache.write_text(
        json.dumps({
            "provenance": provenance,
            "execution": execution_evidence,
            "correct": [1],
        }),
        encoding="utf-8",
    )
    if load_cached_correct(cache, provenance, 1) != [1]:
        failures.append("matching cache provenance was not reused")
    if load_cached_correct(cache, strict_provenance, 1) is not None:
        failures.append("strict protocol reused a generic protocol cache")

    mismatches = {
        "model": "other/model",
        "revision": "revision-b",
        "dtype": "bfloat16",
        "mmlu": 2,
        "arc": 1,
        "low_layers": 7,
        "arm": "mixed_stq",
        "sampling_scheme": {"mmlu": {"mode": "stratified_per_subject"}},
        "dataset_revisions": {"cais/mmlu": "other", "allenai/ai2_arc": "other"},
        "ordered_item_fingerprint": item_fingerprint([changed_item]),
    }
    for field, value in mismatches.items():
        contaminated = dict(provenance, **{field: value})
        cache.write_text(
            json.dumps({
                "provenance": contaminated,
                "execution": execution_evidence,
                "correct": [1],
            }),
            encoding="utf-8",
        )
        if load_cached_correct(cache, provenance, 1) is not None:
            failures.append("cache reused despite mismatched %s" % field)

    cache.write_text(json.dumps({"arm": "dense", "correct": [1]}), encoding="utf-8")
    if load_cached_correct(cache, provenance, 1) is not None:
        failures.append("legacy cache without provenance was reused")
    cache.write_text(
        json.dumps({"provenance": provenance, "correct": [1]}), encoding="utf-8"
    )
    if load_cached_correct(cache, provenance, 1) is not None:
        failures.append("cache without arm execution evidence was reused")

strict_bfloat16_provenance = dict(strict_provenance, dtype="bfloat16")
strict_bfloat16_provenance["runtime"] = dict(
    strict_bfloat16_provenance["runtime"], requested_dtype="bfloat16"
)
valid_strict_execution = {
    "requested_dtype": "bfloat16",
    "parameter_elements_by_dtype_before_plan": {"torch.bfloat16": 2},
    "parameter_elements_by_dtype_after_plan": {"torch.bfloat16": 2},
    "plan_stats": None,
}
with tempfile.TemporaryDirectory() as tmp:
    cache = Path(tmp) / "correct_dense.json"
    valid_strict_cache = {
        "run_id": "completed-run-id",
        "status": "complete",
        "provenance": strict_bfloat16_provenance,
        "execution": valid_strict_execution,
        "correct": [1],
    }
    cache.write_text(json.dumps(valid_strict_cache), encoding="utf-8")
    if load_cached_result(cache, strict_bfloat16_provenance, 1) is None:
        failures.append("strict cache with BF16 execution evidence was rejected")

    for label, metadata in (
        ("failed status", {"run_id": "failed-run-id", "status": "failed"}),
        ("running status", {"run_id": "running-run-id", "status": "running"}),
        ("missing status", {"run_id": "missing-status-run-id"}),
        ("missing run_id", {"status": "complete"}),
        ("empty run_id", {"run_id": "", "status": "complete"}),
        ("blank run_id", {"run_id": "   ", "status": "complete"}),
    ):
        incomplete = {
            "provenance": strict_bfloat16_provenance,
            "execution": valid_strict_execution,
            "correct": [1],
            **metadata,
        }
        cache.write_text(json.dumps(incomplete), encoding="utf-8")
        if load_cached_result(cache, strict_bfloat16_provenance, 1) is not None:
            failures.append("strict cache reused otherwise-valid %s metadata" % label)

    for field, value in {
        "cache_schema": required_new_api["STRICT_CACHE_SCHEMA"] - 1,
        "imatrix": dict(strict_bfloat16_provenance["imatrix"], sha256="0" * 64),
        "source_sha256": dict(strict_bfloat16_provenance["source_sha256"],
                              **{"eval_tasks.py": "0" * 64}),
    }.items():
        contaminated = dict(strict_bfloat16_provenance, **{field: value})
        cache.write_text(json.dumps({
            "provenance": contaminated,
            "execution": valid_strict_execution,
            "correct": [1],
        }), encoding="utf-8")
        if load_cached_result(cache, strict_bfloat16_provenance, 1) is not None:
            failures.append("strict cache reused mismatched %s" % field)

    invalid_strict_executions = [
        dict(
            valid_strict_execution,
            parameter_elements_by_dtype_before_plan={"torch.float16": 2},
        ),
        dict(
            valid_strict_execution,
            parameter_elements_by_dtype_after_plan={"torch.float16": 2},
        ),
        dict(valid_strict_execution, parameter_elements_by_dtype_before_plan=["torch.bfloat16"]),
        dict(valid_strict_execution, parameter_elements_by_dtype_after_plan=None),
    ]
    for invalid_execution in invalid_strict_executions:
        cache.write_text(
            json.dumps({
                "provenance": strict_bfloat16_provenance,
                "execution": invalid_execution,
                "correct": [1],
            }),
            encoding="utf-8",
        )
        try:
            cached = load_cached_result(cache, strict_bfloat16_provenance, 1)
        except (KeyError, TypeError, RuntimeError) as exc:
            failures.append("malformed strict cache evidence crashed: %s" % exc)
            continue
        if cached is not None:
            failures.append("strict cache reused non-BF16 or malformed execution evidence")


class StubAutoTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return object()


class CacheStubModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))

    def eval(self):
        return self

    def parameters(self):
        return iter([self.weight])


model_loads = []


class StubAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        model_loads.append((args, kwargs))
        return CacheStubModel()


class CountingTokenizer:
    calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.calls += 1
        return object()


class CountingModel:
    calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.calls += 1
        return CacheStubModel()


validation_originals = {
    "AutoTokenizer": eval_tasks.AutoTokenizer,
    "AutoModelForCausalLM": eval_tasks.AutoModelForCausalLM,
    "load_mmlu": eval_tasks.load_mmlu,
    "load_mmlu_stratified": eval_tasks.load_mmlu_stratified,
    "load_arc": eval_tasks.load_arc,
    "item_fingerprint": eval_tasks.item_fingerprint,
    "run_arm": eval_tasks.run_arm,
    "torch_load": eval_tasks.torch.load,
}

with tempfile.TemporaryDirectory() as tmp:
    CountingTokenizer.calls = 0
    CountingModel.calls = 0
    imatrix_calls = []
    eval_tasks.AutoTokenizer = CountingTokenizer
    eval_tasks.AutoModelForCausalLM = CountingModel
    eval_tasks.load_mmlu = lambda _limit: [item]
    eval_tasks.load_arc = lambda _limit: []
    eval_tasks.run_arm = lambda *_args: [1]
    eval_tasks.torch.load = lambda *_args, **_kwargs: imatrix_calls.append(1) or {}
    original_argv = sys.argv
    sys.argv = [
        "eval_tasks.py",
        "--model", "org/model",
        "--revision", "revision-a",
        "--imatrix", str(Path(tmp) / "imatrix.json"),
        "--mmlu", "2",
        "--arc", "0",
        "--arms", "dense",
        "--out", str(Path(tmp) / "partial-report.json"),
    ]
    try:
        eval_tasks.main()
    except RuntimeError as exc:
        if "expected 2 MMLU items" not in str(exc):
            failures.append("partial MMLU load raised an unclear error: %s" % exc)
    else:
        failures.append("partial MMLU load was accepted")
    finally:
        sys.argv = original_argv
    if CountingTokenizer.calls or CountingModel.calls or imatrix_calls:
        failures.append("partial item load reached paid tokenizer/imatrix/model loading")

with tempfile.TemporaryDirectory() as tmp:
    CountingTokenizer.calls = 0
    CountingModel.calls = 0
    imatrix_calls = []
    arc_item = dict(item, task="arc_challenge", subject="arc")
    eval_tasks.AutoTokenizer = CountingTokenizer
    eval_tasks.AutoModelForCausalLM = CountingModel
    eval_tasks.load_mmlu_stratified = lambda _limit: [item] * 570
    eval_tasks.load_arc = lambda _limit: [arc_item] * 230
    eval_tasks.item_fingerprint = lambda _items: "wrong-fingerprint"
    eval_tasks.torch.load = lambda *_args, **_kwargs: imatrix_calls.append(1) or {}
    original_argv = sys.argv
    sys.argv = ["eval_tasks.py", *strict_cli, "--out", str(Path(tmp) / "strict-report.json")]
    try:
        eval_tasks.main()
    except RuntimeError as exc:
        if STRICT_PROTOCOL_FINGERPRINT not in str(exc):
            failures.append("strict fingerprint error omitted the required fingerprint")
    else:
        failures.append("strict protocol accepted the wrong ordered item fingerprint")
    finally:
        sys.argv = original_argv
    if CountingTokenizer.calls or CountingModel.calls or imatrix_calls:
        failures.append("strict fingerprint failure reached paid tokenizer/imatrix/model loading")

eval_tasks.AutoTokenizer = validation_originals["AutoTokenizer"]
eval_tasks.AutoModelForCausalLM = validation_originals["AutoModelForCausalLM"]
eval_tasks.load_mmlu = validation_originals["load_mmlu"]
eval_tasks.load_mmlu_stratified = validation_originals["load_mmlu_stratified"]
eval_tasks.load_arc = validation_originals["load_arc"]
eval_tasks.item_fingerprint = validation_originals["item_fingerprint"]
eval_tasks.run_arm = validation_originals["run_arm"]
eval_tasks.torch.load = validation_originals["torch_load"]


originals = {
    "AutoTokenizer": eval_tasks.AutoTokenizer,
    "AutoModelForCausalLM": eval_tasks.AutoModelForCausalLM,
    "load_mmlu": eval_tasks.load_mmlu,
    "load_arc": eval_tasks.load_arc,
    "run_arm": eval_tasks.run_arm,
    "torch_load": eval_tasks.torch.load,
}
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "report.json"
    cache = Path(tmp) / "correct_dense.json"
    (Path(tmp) / "imatrix.pt").write_bytes(b"generic-test-imatrix")
    contaminated = dict(provenance, model="other/model")
    cache.write_text(
        json.dumps({"provenance": contaminated, "correct": [0]}), encoding="utf-8"
    )
    eval_tasks.AutoTokenizer = StubAutoTokenizer
    eval_tasks.AutoModelForCausalLM = StubAutoModel
    eval_tasks.load_mmlu = lambda _limit: [item]
    eval_tasks.load_arc = lambda _limit: []
    eval_tasks.run_arm = lambda *_args: [1]
    eval_tasks.torch.load = lambda *_args, **_kwargs: {}
    argv = [
        "eval_tasks.py",
        "--model", "org/model",
        "--revision", "revision-a",
        "--imatrix", str(Path(tmp) / "imatrix.json"),
        "--mmlu", "1",
        "--arc", "0",
        "--low-layers", "6",
        "--arms", "dense",
        "--dtype", "float16",
        "--out", str(out),
    ]
    original_argv = sys.argv
    try:
        sys.argv = argv
        eval_tasks.main()
        cache_payload = json.loads(cache.read_text(encoding="utf-8"))
        report_payload = json.loads(out.read_text(encoding="utf-8"))
        if len(model_loads) != 1:
            failures.append("mismatched cache did not trigger exactly one recomputation")
        if report_payload["provenance"]["dense"] != cache_payload["provenance"]:
            failures.append("report provenance does not match cache provenance")
        if report_payload["execution"]["dense"] != cache_payload.get("execution"):
            failures.append("report execution evidence does not match cache evidence")
        execution = cache_payload.get("execution", {})
        expected_distribution = {"torch.float16": 2}
        if execution.get("parameter_elements_by_dtype_before_plan") != expected_distribution:
            failures.append("cache is missing the pre-plan parameter dtype distribution")
        if execution.get("parameter_elements_by_dtype_after_plan") != expected_distribution:
            failures.append("cache is missing the post-plan parameter dtype distribution")
        if set(report_payload.get("runtime", {})) != required_runtime_fields:
            failures.append("final report is missing runtime identity")

        eval_tasks.main()
        if len(model_loads) != 1:
            failures.append("matching cache was recomputed instead of reused")
        reused_report = json.loads(out.read_text(encoding="utf-8"))
        if reused_report["execution"]["dense"] != execution:
            failures.append("cache reuse did not propagate arm execution evidence")
    finally:
        sys.argv = original_argv
        eval_tasks.AutoTokenizer = originals["AutoTokenizer"]
        eval_tasks.AutoModelForCausalLM = originals["AutoModelForCausalLM"]
        eval_tasks.load_mmlu = originals["load_mmlu"]
        eval_tasks.load_arc = originals["load_arc"]
        eval_tasks.run_arm = originals["run_arm"]
        eval_tasks.torch.load = originals["torch_load"]


class StrictStubModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    def eval(self):
        return self

    def parameters(self):
        return iter([self.weight])


class StrictAutoModel:
    calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.calls += 1
        return StrictStubModel()


def strict_compare(records, _baseline):
    count = len(records["dense"])
    return {
        "baseline": "dense",
        "items": count,
        "arms": {
            "dense": {"correct": sum(records["dense"]), "accuracy": 1.0},
            "dense_iq3_ref": {
                "correct": sum(records["dense_iq3_ref"]), "accuracy": 1.0,
            },
        },
        "comparisons": {
            "dense_vs_dense_iq3_ref": {
                "only_first_correct": 0,
                "only_second_correct": 0,
                "mcnemar_p": 1.0,
                "accuracy_delta": 0.0,
                "ci_95": [0.0, 0.0],
                "significant": False,
            },
        },
    }


strict_originals = {
    "AutoTokenizer": eval_tasks.AutoTokenizer,
    "AutoModelForCausalLM": eval_tasks.AutoModelForCausalLM,
    "load_mmlu_stratified": eval_tasks.load_mmlu_stratified,
    "load_arc": eval_tasks.load_arc,
    "item_fingerprint": eval_tasks.item_fingerprint,
    "resolve_imatrix_identity": eval_tasks.resolve_imatrix_identity,
    "gpu_preflight": eval_tasks.gpu_preflight,
    "runtime_identity": eval_tasks.runtime_identity,
    "apply_plan": eval_tasks.apply_plan,
    "run_arm": eval_tasks.run_arm,
    "compare": eval_tasks.compare,
    "torch_load": eval_tasks.torch.load,
    "empty_cache": eval_tasks.torch.cuda.empty_cache,
}


def install_strict_main_stubs(imatrix_path, plan_function):
    arc_item = dict(item, task="arc_challenge", subject="arc")
    eval_tasks.AutoTokenizer = StubAutoTokenizer
    eval_tasks.AutoModelForCausalLM = StrictAutoModel
    eval_tasks.load_mmlu_stratified = lambda _limit: [item] * 570
    eval_tasks.load_arc = lambda _limit: [arc_item] * 230
    eval_tasks.item_fingerprint = lambda _items: STRICT_PROTOCOL_FINGERPRINT
    eval_tasks.resolve_imatrix_identity = lambda _path, strict=False: {
        "path": str(imatrix_path.resolve()),
        "size": 7137641,
        "sha256": "def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0",
    }
    eval_tasks.gpu_preflight = lambda: {
        "gpu": {
            "index": 0,
            "name": "NVIDIA H100 NVL",
            "driver": "570.86.15",
            "total_memory_mib": 97871,
            "free_memory_mib": 96800,
        },
        "compute_processes": [],
    }
    eval_tasks.runtime_identity = lambda device, requested_dtype: {
        "platform": "test",
        "python": "test",
        "gpu": {"name": "NVIDIA H100 NVL", "capability": [9, 0]},
        "torch": "test",
        "cuda": "test",
        "transformers": "test",
        "datasets": "test",
        "device": device,
        "requested_dtype": requested_dtype,
    }
    eval_tasks.apply_plan = plan_function
    eval_tasks.compare = strict_compare
    eval_tasks.torch.load = lambda *_args, **_kwargs: {}
    eval_tasks.torch.cuda.empty_cache = lambda: None


def strict_stub_cache_provenance(imatrix_path, arm):
    runtime = eval_tasks.runtime_identity("cuda", "bfloat16")
    runtime["gpu_preflight"] = eval_tasks.gpu_preflight()
    return cache_provenance(
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "bfloat16",
        570,
        230,
        6,
        arm,
        "qwen38_bf16_800",
        {
            "mmlu": {
                "mode": "stratified_per_subject",
                "per_subject": 10,
                "subjects": 57,
                "count": 570,
            },
            "arc": {"mode": "valid_4_choice_prefix", "count": 230},
        },
        STRICT_PROTOCOL_FINGERPRINT,
        runtime,
        imatrix_identity={
            "path": str(imatrix_path.resolve()),
            "size": 7137641,
            "sha256": "def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0",
        },
        source_hashes=eval_tasks.source_identity(),
    )


try:
    for stale_artifact in ("result", "completion"):
        with tempfile.TemporaryDirectory() as tmp:
            StrictAutoModel.calls = 0
            out = Path(tmp) / "strict-existing.json"
            completion = out.with_name(out.name + ".complete.json")
            stale_path = out if stale_artifact == "result" else completion
            stale_bytes = ("stale-%s-must-be-preserved\n" % stale_artifact).encode("utf-8")
            stale_path.write_bytes(stale_bytes)
            imatrix_path = Path(tmp) / "qwen38_imatrix.pt"
            imatrix_path.write_bytes(b"stub")
            install_strict_main_stubs(
                imatrix_path, lambda *_args, **_kwargs: complete_strict_plan_stats())
            eval_tasks.run_arm = lambda _model, _tokenizer, items, _device, _callback=None: (
                [1] * len(items)
            )
            original_argv = sys.argv
            sys.argv = ["eval_tasks.py", *strict_cli, "--out", str(out)]
            try:
                eval_tasks.main()
            except FileExistsError as exc:
                if str(stale_path) not in str(exc):
                    failures.append("strict rerun refusal omitted the existing %s path" % (
                        stale_artifact))
            else:
                failures.append("strict rerun overwrote an existing %s" % stale_artifact)
            finally:
                sys.argv = original_argv
            if stale_path.read_bytes() != stale_bytes:
                failures.append("strict rerun did not preserve the existing %s bytes" % (
                    stale_artifact))
            if out.with_name(out.name + ".progress.json").exists() or out.with_name(
                    out.name + ".failure.json").exists():
                failures.append("strict rerun refusal created progress/failure artifacts")
            if StrictAutoModel.calls:
                failures.append("strict rerun refusal reached model loading")
            if list(Path(tmp).glob(stale_path.name + ".quarantine-*")):
                failures.append("strict rerun quarantined an existing %s" % stale_artifact)

    with tempfile.TemporaryDirectory() as tmp:
        StrictAutoModel.calls = 0
        out = Path(tmp) / "strict-result.json"
        imatrix_path = Path(tmp) / "qwen38_imatrix.pt"
        imatrix_path.write_bytes(b"stub")
        install_strict_main_stubs(
            imatrix_path, lambda *_args, **_kwargs: complete_strict_plan_stats())
        (Path(tmp) / "correct_dense.json").write_text(json.dumps({
            "run_id": "failed-cache-run",
            "status": "failed",
            "provenance": strict_stub_cache_provenance(imatrix_path, "dense"),
            "execution": {
                "requested_dtype": "bfloat16",
                "parameter_elements_by_dtype_before_plan": {"torch.bfloat16": 2},
                "parameter_elements_by_dtype_after_plan": {"torch.bfloat16": 2},
                "plan_stats": None,
            },
            "correct": [1] * 800,
        }), encoding="utf-8")

        def complete_run_arm(_model, _tokenizer, items, _device, on_progress=None):
            if on_progress is not None:
                on_progress(400)
                on_progress(len(items))
            return [1] * len(items)

        eval_tasks.run_arm = complete_run_arm
        original_argv = sys.argv
        sys.argv = ["eval_tasks.py", *strict_cli, "--out", str(out)]
        try:
            eval_tasks.main()
        finally:
            sys.argv = original_argv
        result = json.loads(out.read_text(encoding="utf-8"))
        completion_path = out.with_name(out.name + ".complete.json")
        progress_path = out.with_name(out.name + ".progress.json")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("decision", {}).get(
                "primary") != "noninferior":
            failures.append("fresh strict path did not produce a complete frozen decision")
        if completion.get("result_sha256") != hashlib.sha256(out.read_bytes()).hexdigest():
            failures.append("strict completion marker did not bind the final result SHA-256")
        if progress_payload.get("completed_items") != {
                "dense": 800, "dense_iq3_ref": 800}:
            failures.append("fresh strict progress did not persist per-arm item counts")
        if result.get("execution", {}).get("dense_iq3_ref", {}).get(
                "plan_stats", {}).get("params") != 17112760320:
            failures.append("fresh strict result omitted canonical plan evidence")
        if not list(Path(tmp).glob("correct_dense.json.quarantine-*")):
            failures.append("strict path did not quarantine a previous-schema cache")
        if StrictAutoModel.calls != 2:
            failures.append("fresh strict path did not evaluate exactly two arms")

    with tempfile.TemporaryDirectory() as tmp:
        StrictAutoModel.calls = 0
        out = Path(tmp) / "strict-failure.json"
        imatrix_path = Path(tmp) / "qwen38_imatrix.pt"
        imatrix_path.write_bytes(b"stub")

        def mixed_dtype_plan(model, *_args, **_kwargs):
            model.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
            return complete_strict_plan_stats()

        install_strict_main_stubs(imatrix_path, mixed_dtype_plan)

        def tracked_run_arm(_model, _tokenizer, items, _device, on_progress=None):
            if on_progress is not None:
                on_progress(len(items))
            return [1] * len(items)

        eval_tasks.run_arm = tracked_run_arm
        original_argv = sys.argv
        sys.argv = ["eval_tasks.py", *strict_cli, "--out", str(out)]
        try:
            eval_tasks.main()
        except RuntimeError as exc:
            if "strict BF16 arm dense_iq3_ref" not in str(exc):
                failures.append("post-plan BF16 failure was unclear: %s" % exc)
        else:
            failures.append("fresh strict path accepted a mixed post-plan dtype")
        finally:
            sys.argv = original_argv
        failure_path = out.with_name(out.name + ".failure.json")
        failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure_payload.get("status") != "failed":
            failures.append("caught strict failure did not persist failed status")
        if failure_payload.get("completed_arms") != ["dense"] or failure_payload.get(
                "completed_items") != {"dense": 800, "dense_iq3_ref": 0}:
            failures.append("caught strict failure lost completed arm/item counts")
        if out.exists() or out.with_name(out.name + ".complete.json").exists():
            failures.append("caught strict failure presented a partial result as complete")
finally:
    eval_tasks.AutoTokenizer = strict_originals["AutoTokenizer"]
    eval_tasks.AutoModelForCausalLM = strict_originals["AutoModelForCausalLM"]
    eval_tasks.load_mmlu_stratified = strict_originals["load_mmlu_stratified"]
    eval_tasks.load_arc = strict_originals["load_arc"]
    eval_tasks.item_fingerprint = strict_originals["item_fingerprint"]
    eval_tasks.resolve_imatrix_identity = strict_originals["resolve_imatrix_identity"]
    eval_tasks.gpu_preflight = strict_originals["gpu_preflight"]
    eval_tasks.runtime_identity = strict_originals["runtime_identity"]
    eval_tasks.apply_plan = strict_originals["apply_plan"]
    eval_tasks.run_arm = strict_originals["run_arm"]
    eval_tasks.compare = strict_originals["compare"]
    eval_tasks.torch.load = strict_originals["torch_load"]
    eval_tasks.torch.cuda.empty_cache = strict_originals["empty_cache"]

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)

print("PASS: task scoring verified offline")
print("  prompt renders all four options and the answer cue")
print("  argmax over per-letter log-likelihood picks the favored option in all 4 cases")
print("  MMLU stratifies 10 each across 57 subjects; ARC selects 230 valid items")
print("  dataset revisions are pinned and ambiguous MMLU CLI is rejected")
print("  partial/zero loads fail before paid loading and invalid answers are rejected")
print("  strict protocol tuple and ordered fingerprint are enforced before paid loading")
print("  runtime identity and per-arm dtype evidence persist across cache reuse")
print("  protocol-bound caches and strict BF16 cache evidence are enforced")
print("  cache reuse requires exact provenance and deterministic item fingerprint")
print("  mismatched cache recomputes; report and cache provenance match")
print("  no GPU, no network, no model download required")
