"""Strict 800-item llama-server Top-1 evaluation for the pinned Qwen3.8-27B GGUF arms."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

PROTOCOL = "qwen38_gguf_800_llama_server"
LETTERS = ("A", "B", "C", "D")
ARMS = ("BF16", "IQ3_XXS", "IQ4_XS", "Q4_K_M", "Q5_K_M")
LLAMA_CPP_COMMIT = "580e88d8b7dece7099d9b62323521d0254ff3615"
ITEM_FINGERPRINT = "a72515282c6fc20f34188b3102d99468ab2b02266105ed9c6e4ec405fbad8fd0"
EXPECTED_MMLU_PER_SUBJECT = 10
EXPECTED_MMLU_ITEMS = 570
EXPECTED_ARC_ITEMS = 230
COMPLETION_SEED = 22
LETTER_BIAS = 100.0
DEFAULT_TIMEOUT_SECONDS = 600.0
SHA256_CHUNK_BYTES = 1024 * 1024
MODEL_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
TOP_PROB_KEYS = ("top_logprobs", "top_probs", "probs")

_EVAL_TASKS_MODULE = None


class EvaluationError(ValueError):
    pass


def _eval_tasks():
    """Import the shared 800-item task surface lazily so the CLI stays importable offline."""

    global _EVAL_TASKS_MODULE
    if _EVAL_TASKS_MODULE is None:
        # eval_tasks and its dependencies use flat module names, exactly like the
        # repository test harness, so the package directory must be importable.
        source_dir = str(Path(__file__).resolve().parent)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        try:
            _EVAL_TASKS_MODULE = importlib.import_module("eval_tasks")
        except ImportError as error:
            raise EvaluationError(
                "eval_tasks requires the pinned torch, transformers, and datasets extras"
            ) from error
    return _EVAL_TASKS_MODULE


def artifact_paths(out: Path) -> dict[str, Path]:
    out = Path(out)
    return {
        "result": out,
        "completion": out.with_name(out.name + ".complete.json"),
        "progress": out.with_name(out.name + ".progress.json"),
        "failure": out.with_name(out.name + ".failure.json"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_json(path: Path, payload: Mapping[str, object]) -> Path:
    data = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = _stage_json(path, payload)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish an immutable artifact; an existing path is never overwritten."""

    temporary = _stage_json(path, payload)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise EvaluationError(f"refusing to overwrite an existing artifact: {path}") from error
        except OSError as error:
            raise EvaluationError(f"artifact could not be published: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_server(server: object) -> str:
    if not isinstance(server, str) or not server.strip():
        raise EvaluationError("server URL must be a nonempty string")
    normalized = server.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise EvaluationError("server URL must be an http or https origin")
    if parsed.username or parsed.password:
        raise EvaluationError("server URL must not carry credentials")
    if parsed.path or parsed.query or parsed.fragment:
        raise EvaluationError("server URL must be an origin without path, query, or fragment")
    return normalized


def _validate_identity(arm: object, model_sha256: object, llama_commit: object) -> tuple[str, ...]:
    if arm not in ARMS:
        raise EvaluationError(f"arm must be one of {', '.join(ARMS)}")
    if not isinstance(model_sha256, str) or MODEL_SHA256_PATTERN.fullmatch(model_sha256) is None:
        raise EvaluationError("model SHA-256 must be 64 lowercase hexadecimal characters")
    if llama_commit != LLAMA_CPP_COMMIT:
        raise EvaluationError(f"llama.cpp commit must be the pinned {LLAMA_CPP_COMMIT}")
    return arm, model_sha256, llama_commit


def _validate_timeout(timeout_seconds: object) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise EvaluationError("timeout must be numeric")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise EvaluationError("timeout must be finite and positive")
    return timeout


def _post_json(
    server: str, path: str, payload: Mapping[str, object], timeout: float
) -> Mapping[str, object]:
    data = json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        server + path,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise EvaluationError(f"llama-server {path} returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise EvaluationError(f"llama-server {path} request failed: {error}") from error
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"llama-server {path} returned invalid JSON") from error
    if not isinstance(parsed, Mapping):
        raise EvaluationError(f"llama-server {path} returned a non-object response")
    return parsed


def _token_ids(value: object, description: str) -> list[int]:
    if not isinstance(value, list):
        raise EvaluationError(f"llama-server {description} must be a list of token ids")
    ids = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int):
            raise EvaluationError(f"llama-server {description} must contain integer token ids")
        ids.append(int(token))
    return ids


def resolve_letter_tokens(server: str, timeout: float) -> list[int]:
    """Require that each answer letter is exactly one distinct token on this server."""

    letter_ids: list[int] = []
    for letter in LETTERS:
        content = " " + letter
        response = _post_json(server, "/tokenize", {"content": content, "add_special": False},
                              timeout)
        tokens = _token_ids(response.get("tokens"), "tokenization")
        if len(tokens) != 1:
            raise EvaluationError(
                f"llama-server tokenized {content!r} into {len(tokens)} tokens; "
                "each candidate letter must be a single token"
            )
        letter_ids.append(tokens[0])
    if len(set(letter_ids)) != len(LETTERS):
        raise EvaluationError("llama-server letter tokens must be four distinct token ids")
    return letter_ids


def build_completion_payload(prompt: str, letter_ids: Sequence[int]) -> dict[str, object]:
    """Return the exact preregistered completion request for one item."""

    return {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": -1.0,
        "seed": COMPLETION_SEED,
        "cache_prompt": False,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": [[int(token_id), LETTER_BIAS] for token_id in letter_ids],
        "n_probs": len(LETTERS),
        "return_tokens": True,
    }


def request_contract() -> dict[str, object]:
    contract = {
        key: value
        for key, value in build_completion_payload("", range(len(LETTERS))).items()
        if key not in ("prompt", "logit_bias")
    }
    contract["logit_bias_value"] = LETTER_BIAS
    contract["letters"] = list(LETTERS)
    return contract


def _top_token_ids(entry: object) -> list[int]:
    if not isinstance(entry, Mapping):
        raise EvaluationError("llama-server completion probabilities are malformed")
    candidates = None
    for key in TOP_PROB_KEYS:
        value = entry.get(key)
        if isinstance(value, list):
            candidates = value
            break
    if candidates is None:
        raise EvaluationError("llama-server did not return top-4 candidate probabilities")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or isinstance(candidate.get("id"), bool):
            raise EvaluationError("llama-server did not return top-4 candidate token ids")
        token = candidate.get("id")
        if not isinstance(token, int):
            raise EvaluationError("llama-server did not return top-4 candidate token ids")
        ids.append(int(token))
    return ids


def score_completion(response: Mapping[str, object], letter_ids: Sequence[int]) -> dict[str, object]:
    """Validate one strict completion response and return its candidate prediction."""

    tokens = _token_ids(response.get("tokens"), "completion tokens")
    if len(tokens) != 1:
        raise EvaluationError(
            f"llama-server returned {len(tokens)} completion tokens; exactly one is required"
        )
    token = tokens[0]
    if token not in letter_ids:
        raise EvaluationError("llama-server returned a token outside the candidate set")
    prediction = list(letter_ids).index(token)
    probabilities = response.get("completion_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 1:
        raise EvaluationError("llama-server did not return exactly one completion probability set")
    top_ids = _top_token_ids(probabilities[0])
    if sorted(top_ids) != sorted(letter_ids):
        raise EvaluationError("llama-server top-4 token ids are not the candidate set")
    content = response.get("content")
    if isinstance(content, str) and content.strip() != LETTERS[prediction]:
        raise EvaluationError("llama-server completion text does not match its candidate token")
    return {"prediction": prediction, "tokens": tokens, "top_token_ids": top_ids}


def _validate_items(items: Sequence[Mapping[str, object]]) -> None:
    for index, item in enumerate(items):
        choices = item.get("choices") if isinstance(item, Mapping) else None
        answer = item.get("answer") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("task"), str)
            or not isinstance(item.get("subject"), str)
            or not isinstance(item.get("question"), str)
            or not isinstance(choices, list)
            or len(choices) != len(LETTERS)
            or isinstance(answer, bool)
            or not isinstance(answer, int)
            or answer not in range(len(LETTERS))
        ):
            raise EvaluationError(f"item {index} does not match the pinned 800-item schema")


def load_items(mmlu_loader=None, arc_loader=None) -> list[dict[str, object]]:
    """Load the pinned 570 MMLU and 230 ARC items and enforce the ordered fingerprint."""

    tasks = _eval_tasks()
    if mmlu_loader is None:
        mmlu_loader = tasks.load_mmlu_stratified
    if arc_loader is None:
        arc_loader = tasks.load_arc
    try:
        mmlu_items = list(mmlu_loader(EXPECTED_MMLU_PER_SUBJECT))
        arc_items = list(arc_loader(EXPECTED_ARC_ITEMS))
        tasks.validate_item_counts(
            mmlu_items, arc_items, EXPECTED_MMLU_ITEMS, EXPECTED_ARC_ITEMS
        )
    except EvaluationError:
        raise
    except (RuntimeError, ValueError, OSError) as error:
        raise EvaluationError(f"pinned 800-item load failed: {error}") from error
    items = mmlu_items + arc_items
    _validate_items(items)
    fingerprint = tasks.item_fingerprint(items)
    if fingerprint != ITEM_FINGERPRINT:
        raise EvaluationError(
            f"ordered item fingerprint {fingerprint} does not match the preregistered "
            f"{ITEM_FINGERPRINT}"
        )
    return items


def build_provenance(
    arm: str,
    model_sha256: str,
    llama_commit: str,
    fingerprint: str,
    letter_ids: Sequence[int],
    mmlu_items: int,
    arc_items: int,
) -> dict[str, object]:
    tasks = _eval_tasks()
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "model_sha256": model_sha256,
        "llama_commit": llama_commit,
        "ordered_item_fingerprint": fingerprint,
        "items": mmlu_items + arc_items,
        "mmlu": mmlu_items,
        "arc": arc_items,
        "mmlu_per_subject": EXPECTED_MMLU_PER_SUBJECT,
        "dataset_revisions": {
            "cais/mmlu": tasks.MMLU_DATASET_REVISION,
            "allenai/ai2_arc": tasks.ARC_DATASET_REVISION,
        },
        "letter_token_ids": [int(token_id) for token_id in letter_ids],
        "request_contract": request_contract(),
    }


def validate_record(
    record: object,
    item: Mapping[str, object],
    index: int,
    letter_ids: Sequence[int],
) -> dict[str, object]:
    """Validate one item record against the item it claims to answer."""

    if not isinstance(record, Mapping):
        raise EvaluationError(f"item record {index} is not an object")
    prediction = record.get("prediction")
    seconds = record.get("request_seconds")
    if (
        record.get("index") != index
        or record.get("task") != item["task"]
        or record.get("subject") != item["subject"]
        or record.get("answer") != item["answer"]
        or isinstance(prediction, bool)
        or not isinstance(prediction, int)
        or prediction not in range(len(LETTERS))
        or record.get("correct") != int(prediction == item["answer"])
        or record.get("tokens") != [int(letter_ids[prediction])]
        or sorted(_token_ids(record.get("top_token_ids"), "record top token ids"))
        != sorted(int(token_id) for token_id in letter_ids)
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0
    ):
        raise EvaluationError(f"item record {index} does not match the strict record schema")
    return dict(record)


def _load_progress(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"resume progress is not readable JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise EvaluationError(f"resume progress is not an object: {path}")
    return payload


def _resume_records(
    progress: Mapping[str, object],
    provenance: Mapping[str, object],
    items: Sequence[Mapping[str, object]],
    letter_ids: Sequence[int],
) -> tuple[str, list[dict[str, object]]]:
    if progress.get("provenance") != provenance:
        raise EvaluationError("resume refused: stored progress provenance does not match this run")
    run_id = progress.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise EvaluationError("resume refused: stored progress has no run identifier")
    records = progress.get("records")
    if not isinstance(records, list) or len(records) > len(items):
        raise EvaluationError("resume refused: stored progress records are invalid")
    return run_id, [
        validate_record(record, items[index], index, letter_ids)
        for index, record in enumerate(records)
    ]


def _progress_payload(
    run_id: str,
    arm: str,
    provenance: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    status: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "status": status,
        "protocol": PROTOCOL,
        "arm": arm,
        "provenance": dict(provenance),
        "completed_items": len(records),
        "records": [dict(record) for record in records],
    }
    if extra:
        payload.update(extra)
    return payload


def run_evaluation(
    *,
    server: str,
    arm: str,
    model_sha256: str,
    llama_commit: str,
    out: Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mmlu_loader=None,
    arc_loader=None,
) -> dict[str, object]:
    """Score the pinned 800 items against one llama-server arm and publish the result."""

    server = _validate_server(server)
    arm, model_sha256, llama_commit = _validate_identity(arm, model_sha256, llama_commit)
    timeout = _validate_timeout(timeout_seconds)
    paths = artifact_paths(out)
    completed = [name for name in ("result", "completion") if paths[name].exists()]
    if completed:
        raise EvaluationError(
            "refusing to overwrite a completed result: "
            + ", ".join(str(paths[name]) for name in completed)
        )
    state: dict[str, object] = {"run_id": uuid.uuid4().hex, "records": [], "arm": arm}
    try:
        return _execute(
            server=server,
            arm=arm,
            model_sha256=model_sha256,
            llama_commit=llama_commit,
            paths=paths,
            timeout=timeout,
            mmlu_loader=mmlu_loader,
            arc_loader=arc_loader,
            state=state,
        )
    except BaseException as error:
        _record_failure(paths, state, error)
        raise


def _record_failure(paths: Mapping[str, Path], state: Mapping[str, object], error: BaseException):
    run_id = state["run_id"]
    arm = state["arm"]
    records = state["records"]
    provenance = state.get("provenance")
    _write_json_atomic(
        paths["failure"],
        {
            "run_id": run_id,
            "status": "failed",
            "protocol": PROTOCOL,
            "arm": arm,
            "completed_items": len(records),
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    if provenance is not None:
        _write_json_atomic(
            paths["progress"],
            _progress_payload(
                run_id,
                arm,
                provenance,
                records,
                "failed",
                {"error_type": type(error).__name__, "error": str(error)},
            ),
        )


def _execute(
    *,
    server: str,
    arm: str,
    model_sha256: str,
    llama_commit: str,
    paths: Mapping[str, Path],
    timeout: float,
    mmlu_loader,
    arc_loader,
    state: dict[str, object],
) -> dict[str, object]:
    tasks = _eval_tasks()
    items = load_items(mmlu_loader, arc_loader)
    letter_ids = resolve_letter_tokens(server, timeout)
    provenance = build_provenance(
        arm,
        model_sha256,
        llama_commit,
        tasks.item_fingerprint(items),
        letter_ids,
        EXPECTED_MMLU_ITEMS,
        EXPECTED_ARC_ITEMS,
    )
    state["provenance"] = provenance
    progress = _load_progress(paths["progress"])
    records: list[dict[str, object]] = []
    if progress is not None:
        run_id, records = _resume_records(progress, provenance, items, letter_ids)
        state["run_id"] = run_id
    state["records"] = records
    resumed_items = len(records)
    started_epoch = time.time()
    started = time.monotonic()
    _write_json_atomic(
        paths["progress"], _progress_payload(state["run_id"], arm, provenance, records, "running")
    )
    for index in range(resumed_items, len(items)):
        item = items[index]
        payload = build_completion_payload(tasks.render(item), letter_ids)
        request_started = time.monotonic()
        response = _post_json(server, "/completion", payload, timeout)
        elapsed = time.monotonic() - request_started
        scored = score_completion(response, letter_ids)
        prediction = scored["prediction"]
        records.append(
            validate_record(
                {
                    "index": index,
                    "task": item["task"],
                    "subject": item["subject"],
                    "answer": item["answer"],
                    "prediction": prediction,
                    "correct": int(prediction == item["answer"]),
                    "tokens": scored["tokens"],
                    "top_token_ids": scored["top_token_ids"],
                    "request_seconds": elapsed,
                },
                item,
                index,
                letter_ids,
            )
        )
        _write_json_atomic(
            paths["progress"],
            _progress_payload(state["run_id"], arm, provenance, records, "running"),
        )
    if len(records) != len(items):
        raise EvaluationError(
            f"expected {len(items)} item records, produced {len(records)}"
        )
    validated = [
        validate_record(record, items[index], index, letter_ids)
        for index, record in enumerate(records)
    ]
    correct = [int(record["correct"]) for record in validated]
    completed_epoch = time.time()
    result = {
        "run_id": state["run_id"],
        "status": "complete",
        "protocol": PROTOCOL,
        "arm": arm,
        "provenance": provenance,
        "items": len(items),
        "resumed_items": resumed_items,
        "correct": correct,
        "correct_count": sum(correct),
        "accuracy": sum(correct) / len(correct),
        "records": validated,
        "timing": {
            "started_epoch": started_epoch,
            "completed_epoch": completed_epoch,
            "wall_seconds": time.monotonic() - started,
            "request_seconds": sum(float(record["request_seconds"]) for record in validated),
        },
        "runtime": {"server": server, "python": platform.python_version()},
    }
    _publish_json(paths["result"], result)
    _publish_json(
        paths["completion"],
        {
            "run_id": state["run_id"],
            "status": "complete",
            "protocol": PROTOCOL,
            "arm": arm,
            "result": str(Path(paths["result"]).resolve()),
            "result_sha256": sha256_file(paths["result"]),
        },
    )
    _write_json_atomic(
        paths["progress"],
        _progress_payload(state["run_id"], arm, provenance, validated, "complete"),
    )
    return json.loads(Path(paths["result"]).read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None, *, mmlu_loader=None, arc_loader=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(
            server=args.server,
            arm=args.arm,
            model_sha256=args.model_sha256,
            llama_commit=args.llama_commit,
            out=args.out,
            timeout_seconds=args.timeout_seconds,
            mmlu_loader=mmlu_loader,
            arc_loader=arc_loader,
        )
    except EvaluationError as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(
        "%s %d/%d (%.4f)\n"
        % (result["arm"], result["correct_count"], result["items"], result["accuracy"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
