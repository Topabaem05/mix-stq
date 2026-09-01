from __future__ import annotations

import json
import re
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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

import eval_llama_server as evaluator  # noqa: E402
import eval_tasks  # noqa: E402
import pytest  # noqa: E402

MODEL_SHA256 = "b" * 64
LLAMA_COMMIT = evaluator.LLAMA_CPP_COMMIT
LETTER_IDS = (1100, 1101, 1102, 1103)
# Observed pre-sampling, unbiased top-4 from the 2026-09-02 pinned-commit probe.
NATURAL_TOP_IDS = (12095, 1304, 30743, 32671)
NATURAL_TOP_PIECES = (" Paris", " __", " ____", " ______")


def _mmlu_items(count: int) -> list[dict[str, object]]:
    return [
        {
            "task": "mmlu",
            "subject": "subject_%02d" % (index % 57),
            "question": "mmlu question %d" % index,
            "choices": ["alpha", "beta", "gamma", "delta"],
            "answer": index % 4,
        }
        for index in range(count)
    ]


def _arc_items(count: int) -> list[dict[str, object]]:
    return [
        {
            "task": "arc_challenge",
            "subject": "arc",
            "question": "arc question %d" % index,
            "choices": ["alpha", "beta", "gamma", "delta"],
            "answer": (index + 1) % 4,
        }
        for index in range(count)
    ]


def _loaders(mmlu: list[dict[str, object]], arc: list[dict[str, object]]):
    return {
        "mmlu_loader": lambda per_subject: list(mmlu),
        "arc_loader": lambda limit: list(arc),
    }


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    mmlu: list[dict[str, object]],
    arc: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(evaluator, "EXPECTED_MMLU_ITEMS", len(mmlu))
    monkeypatch.setattr(evaluator, "EXPECTED_ARC_ITEMS", len(arc))
    monkeypatch.setattr(
        evaluator, "ITEM_FINGERPRINT", eval_tasks.item_fingerprint(mmlu + arc)
    )


def _completion_body(
    letter_index: int, *, top_ids=NATURAL_TOP_IDS, tokens=None, logprobs=None
) -> dict[str, object]:
    """Mirror the pinned server: the sampled token is biased, top_logprobs is not.

    The default top_logprobs list is the observed pre-sampling, unbiased top-4 from the
    2026-09-02 probe (amendment 1), not the candidate letter set.
    """

    token_id = LETTER_IDS[letter_index]
    if logprobs is None:
        logprobs = [-0.5 - position for position in range(len(top_ids))]
    pieces = [
        NATURAL_TOP_PIECES[position] if position < len(NATURAL_TOP_PIECES) else " piece%d" % position
        for position in range(len(top_ids))
    ]
    return {
        "content": " " + evaluator.LETTERS[letter_index],
        "tokens": [token_id] if tokens is None else list(tokens),
        "completion_probabilities": [
            {
                "id": token_id,
                "token": " " + evaluator.LETTERS[letter_index],
                "logprob": -6.78,
                "top_logprobs": [
                    {"id": candidate, "token": piece, "logprob": logprob}
                    for candidate, piece, logprob in zip(
                        top_ids, pieces, logprobs, strict=True
                    )
                ],
            }
        ],
    }


def _tokenize_body(letter: str, *, ids=None) -> dict[str, object]:
    index = evaluator.LETTERS.index(letter.strip())
    return {"tokens": [LETTER_IDS[index]] if ids is None else list(ids)}


def _default_responder(path: str, body: dict[str, object]):
    if path == "/tokenize":
        return 200, _tokenize_body(str(body["content"]))
    return 200, _completion_body(0)


def _fail_on_third_item(path: str, body: dict[str, object]):
    if path == "/completion" and str(body["prompt"]).startswith("mmlu question 2"):
        raise _FakeServerFailure("induced protocol failure")
    return _default_responder(path, body)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append((self.path, body))
        try:
            status, payload = self.server.responder(self.path, body)
        except _FakeServerFailure as failure:
            status, payload = 500, {"error": str(failure)}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


class _FakeServerFailure(Exception):
    pass


class _FakeServer:
    def __init__(self, responder=_default_responder) -> None:
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._http.responder = responder
        self._http.requests = []
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._http.server_address[:2]
        return "http://%s:%d" % (host, port)

    @property
    def requests(self) -> list[tuple[str, dict[str, object]]]:
        return self._http.requests

    def completions(self) -> list[dict[str, object]]:
        return [body for path, body in self.requests if path == "/completion"]

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server():
    running = _FakeServer()
    try:
        yield running
    finally:
        running.close()


def _run(server_url: str, out: Path, mmlu, arc, **overrides) -> dict[str, object]:
    kwargs = {
        "server": server_url,
        "arm": "BF16",
        "model_sha256": MODEL_SHA256,
        "llama_commit": LLAMA_COMMIT,
        "out": out,
        **_loaders(mmlu, arc),
    }
    kwargs.update(overrides)
    return evaluator.run_evaluation(**kwargs)


def test_completion_requests_use_the_strict_sampling_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    _run(server.url, tmp_path / "bf16.json", mmlu, arc)

    tokenize = [body for path, body in server.requests if path == "/tokenize"]
    assert [body["content"] for body in tokenize] == [" A", " B", " C", " D"]
    requests = server.completions()
    assert len(requests) == 5
    request = requests[0]
    assert request["prompt"] == eval_tasks.render(mmlu[0])
    assert request["n_predict"] == 1
    assert request["temperature"] == -1.0
    assert request["seed"] == 22
    assert request["cache_prompt"] is False
    assert request["repeat_penalty"] == 1.0
    assert request["presence_penalty"] == 0.0
    assert request["frequency_penalty"] == 0.0
    assert sorted(bias for _, bias in request["logit_bias"]) == [100.0] * 4
    assert [token for token, _ in request["logit_bias"]] == list(LETTER_IDS)
    assert request["n_probs"] == 4
    assert request["return_tokens"] is True
    assert set(request) == {
        "prompt",
        "n_predict",
        "temperature",
        "seed",
        "cache_prompt",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "n_probs",
        "return_tokens",
    }


def test_publishes_an_immutable_result_and_completion_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "bf16.json"

    result = _run(server.url, out, mmlu, arc)
    paths = evaluator.artifact_paths(out)

    expected = [int(item["answer"] == 0) for item in mmlu + arc]
    assert result["correct"] == expected
    assert result["correct_count"] == sum(expected)
    assert len(result["records"]) == 5
    assert result["provenance"]["letter_token_ids"] == list(LETTER_IDS)
    assert result["provenance"]["model_sha256"] == MODEL_SHA256
    assert result["records"][0]["tokens"] == [LETTER_IDS[0]]
    assert result["timing"]["wall_seconds"] >= 0.0

    published = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert published == result
    marker = json.loads(paths["completion"].read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["run_id"] == result["run_id"]
    assert marker["result_sha256"] == evaluator.sha256_file(paths["result"])
    progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["completed_items"] == 5


def test_full_eight_hundred_item_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(570), _arc_items(230)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "iq3.json"

    result = _run(server.url, out, mmlu, arc, arm="IQ3_XXS")

    assert len(server.completions()) == 800
    assert len(result["records"]) == 800
    assert len(result["correct"]) == 800
    assert result["provenance"]["items"] == 800
    assert result["provenance"]["mmlu"] == 570
    assert result["provenance"]["arc"] == 230
    assert evaluator.artifact_paths(out)["completion"].is_file()


def test_letters_must_be_four_distinct_single_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    def multi_token(path: str, body: dict[str, object]):
        if path == "/tokenize" and body["content"] == " C":
            return 200, {"tokens": [77, 78]}
        return _default_responder(path, body)

    running = _FakeServer(multi_token)
    try:
        with pytest.raises(evaluator.EvaluationError, match="single token"):
            _run(running.url, tmp_path / "a.json", mmlu, arc)
        assert running.completions() == []
    finally:
        running.close()

    def duplicate(path: str, body: dict[str, object]):
        if path == "/tokenize":
            return 200, {"tokens": [LETTER_IDS[0]]}
        return _default_responder(path, body)

    running = _FakeServer(duplicate)
    try:
        with pytest.raises(evaluator.EvaluationError, match="distinct"):
            _run(running.url, tmp_path / "b.json", mmlu, arc)
        assert running.completions() == []
    finally:
        running.close()


def test_rejects_a_completion_token_outside_the_candidate_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    def outsider(path: str, body: dict[str, object]):
        if path == "/completion":
            return 200, _completion_body(0, tokens=[4242])
        return _default_responder(path, body)

    running = _FakeServer(outsider)
    out = tmp_path / "outsider.json"
    try:
        with pytest.raises(evaluator.EvaluationError, match="candidate"):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()
    assert not evaluator.artifact_paths(out)["result"].exists()


def test_records_the_natural_pre_sampling_top_four_without_rejecting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    """Amendment 1: top_logprobs is pre-sampling and unbiased, so it is recorded, not gated."""

    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "natural.json"

    result = _run(server.url, out, mmlu, arc)

    assert len(result["records"]) == 5
    for record in result["records"]:
        assert record["pre_sampling_top_ids"] == list(NATURAL_TOP_IDS)
        assert record["pre_sampling_top_logprobs"] == [-0.5, -1.5, -2.5, -3.5]
        assert record["tokens"] == [LETTER_IDS[0]]
    assert sorted(NATURAL_TOP_IDS) != sorted(LETTER_IDS)
    assert evaluator.artifact_paths(out)["completion"].is_file()


def test_records_a_null_diagnostic_logprob_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    def with_null(path: str, body: dict[str, object]):
        if path == "/completion":
            return 200, _completion_body(0, logprobs=[-1.25, None, -2.5, -3.5])
        return _default_responder(path, body)

    running = _FakeServer(with_null)
    try:
        result = _run(running.url, tmp_path / "null-logprob.json", mmlu, arc)
    finally:
        running.close()

    assert result["records"][0]["pre_sampling_top_logprobs"] == [-1.25, None, -2.5, -3.5]


def _reject_top_logprobs_arity(tmp_path: Path, mmlu, arc, top_ids, name: str) -> None:
    """The pinned contract requires exactly four diagnostic entries; any other arity stops."""

    def wrong_arity(path: str, body: dict[str, object]):
        if path == "/completion":
            return 200, _completion_body(0, top_ids=top_ids)
        return _default_responder(path, body)

    running = _FakeServer(wrong_arity)
    out = tmp_path / name
    try:
        with pytest.raises(evaluator.EvaluationError, match="requires exactly 4"):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()

    paths = evaluator.artifact_paths(out)
    assert not paths["result"].exists()
    assert not paths["completion"].exists()


def test_rejects_an_empty_top_logprobs_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    _reject_top_logprobs_arity(tmp_path, mmlu, arc, (), "empty.json")


def test_rejects_a_single_entry_top_logprobs_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    # The collapsed post_sampling_probs shape observed in the 2026-09-02 probe.
    _reject_top_logprobs_arity(tmp_path, mmlu, arc, (LETTER_IDS[0],), "single.json")


def test_rejects_a_seven_entry_top_logprobs_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    _reject_top_logprobs_arity(
        tmp_path, mmlu, arc, (*NATURAL_TOP_IDS, 5001, 5002, 5003), "seven.json"
    )


def test_candidate_set_top_four_is_still_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    def biased(path: str, body: dict[str, object]):
        if path == "/completion":
            return 200, _completion_body(0, top_ids=LETTER_IDS)
        return _default_responder(path, body)

    running = _FakeServer(biased)
    try:
        result = _run(running.url, tmp_path / "biased-top4.json", mmlu, arc)
    finally:
        running.close()

    assert result["records"][0]["pre_sampling_top_ids"] == list(LETTER_IDS)


def test_rejects_a_response_without_the_pinned_top_logprobs_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    def renamed(path: str, body: dict[str, object]):
        if path == "/completion":
            payload = _completion_body(0)
            entry = payload["completion_probabilities"][0]
            entry["probs"] = entry.pop("top_logprobs")
            return 200, payload
        return _default_responder(path, body)

    running = _FakeServer(renamed)
    out = tmp_path / "renamed.json"
    try:
        with pytest.raises(evaluator.EvaluationError, match="top_logprobs"):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()
    assert not evaluator.artifact_paths(out)["result"].exists()


def test_wrong_fingerprint_is_rejected_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    monkeypatch.setattr(evaluator, "EXPECTED_MMLU_ITEMS", len(mmlu))
    monkeypatch.setattr(evaluator, "EXPECTED_ARC_ITEMS", len(arc))
    out = tmp_path / "bad-fingerprint.json"

    with pytest.raises(evaluator.EvaluationError, match="fingerprint"):
        _run(server.url, out, mmlu, arc)

    assert server.requests == []
    assert not evaluator.artifact_paths(out)["progress"].exists()
    assert not evaluator.artifact_paths(out)["result"].exists()


def test_wrong_item_counts_are_rejected_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    with pytest.raises(evaluator.EvaluationError, match="items"):
        _run(server.url, tmp_path / "counts.json", mmlu[:2], arc)

    assert server.requests == []


def test_partial_progress_is_not_promoted_and_resume_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "resume.json"
    paths = evaluator.artifact_paths(out)

    running = _FakeServer(_fail_on_third_item)
    try:
        with pytest.raises(evaluator.EvaluationError):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()

    assert not paths["result"].exists()
    assert not paths["completion"].exists()
    progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["completed_items"] == 2
    assert len(progress["records"]) == 2
    assert paths["failure"].is_file()

    resumed = _FakeServer()
    try:
        result = _run(resumed.url, out, mmlu, arc)
        assert len(resumed.completions()) == 3
    finally:
        resumed.close()

    assert result["resumed_items"] == 2
    assert len(result["records"]) == 5
    assert result["run_id"] == progress["run_id"]
    assert paths["completion"].is_file()


def test_resume_rejects_a_provenance_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "mismatch.json"
    paths = evaluator.artifact_paths(out)

    running = _FakeServer(_fail_on_third_item)
    try:
        with pytest.raises(evaluator.EvaluationError):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()

    stored = paths["progress"].read_text(encoding="utf-8")

    resumed = _FakeServer()
    try:
        with pytest.raises(evaluator.EvaluationError, match="provenance"):
            _run(resumed.url, out, mmlu, arc, model_sha256="c" * 64)
        assert resumed.completions() == []
    finally:
        resumed.close()

    assert not paths["result"].exists()
    assert not paths["completion"].exists()
    progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
    assert paths["progress"].read_text(encoding="utf-8") == stored
    assert progress["provenance"]["model_sha256"] == MODEL_SHA256
    assert progress["completed_items"] == 2
    assert len(progress["records"]) == 2


def test_refused_resume_preserves_the_stored_progress_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "preserved.json"
    paths = evaluator.artifact_paths(out)

    running = _FakeServer(_fail_on_third_item)
    try:
        with pytest.raises(evaluator.EvaluationError):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()

    stored = paths["progress"].read_text(encoding="utf-8")
    before = json.loads(stored)
    assert before["completed_items"] == 2
    assert before["provenance"]["model_sha256"] == MODEL_SHA256

    refused = _FakeServer()
    try:
        with pytest.raises(evaluator.EvaluationError, match="provenance"):
            _run(refused.url, out, mmlu, arc, model_sha256="c" * 64)
    finally:
        refused.close()

    assert paths["progress"].read_text(encoding="utf-8") == stored
    after = json.loads(paths["progress"].read_text(encoding="utf-8"))
    assert after["provenance"] == before["provenance"]
    assert after["records"] == before["records"]
    assert after["run_id"] == before["run_id"]
    assert not paths["result"].exists()
    assert not paths["completion"].exists()

    resumed = _FakeServer()
    try:
        result = _run(resumed.url, out, mmlu, arc)
        assert len(resumed.completions()) == 3
    finally:
        resumed.close()

    assert result["run_id"] == before["run_id"]
    assert result["resumed_items"] == 2
    assert len(result["records"]) == 5
    assert result["records"][:2] == before["records"]
    assert paths["completion"].is_file()


def test_completed_run_supersedes_its_own_failure_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "superseded.json"
    paths = evaluator.artifact_paths(out)

    running = _FakeServer(_fail_on_third_item)
    try:
        with pytest.raises(evaluator.EvaluationError):
            _run(running.url, out, mmlu, arc)
    finally:
        running.close()

    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    failed_run_id = failure["run_id"]

    resumed = _FakeServer()
    try:
        result = _run(resumed.url, out, mmlu, arc)
    finally:
        resumed.close()

    assert result["run_id"] == failed_run_id
    assert result["status"] == "complete"
    assert not paths["failure"].exists()
    assert json.loads(paths["progress"].read_text(encoding="utf-8"))["status"] == "complete"


def test_a_foreign_failure_artifact_is_not_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "foreign.json"
    paths = evaluator.artifact_paths(out)
    foreign = {"run_id": "another-run", "status": "failed"}
    paths["failure"].write_text(json.dumps(foreign), encoding="utf-8")

    _run(server.url, out, mmlu, arc)

    assert json.loads(paths["failure"].read_text(encoding="utf-8")) == foreign


def test_refuses_to_overwrite_a_completed_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "done.json"

    first = _run(server.url, out, mmlu, arc)
    before = evaluator.artifact_paths(out)["result"].read_text(encoding="utf-8")

    with pytest.raises(evaluator.EvaluationError, match="completed result"):
        _run(server.url, out, mmlu, arc)

    assert evaluator.artifact_paths(out)["result"].read_text(encoding="utf-8") == before
    assert len(server.completions()) == 5
    assert first["status"] == "complete"


def test_rejects_invalid_identity_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)

    with pytest.raises(evaluator.EvaluationError, match="arm"):
        _run(server.url, tmp_path / "a.json", mmlu, arc, arm="Q6_K")
    with pytest.raises(evaluator.EvaluationError, match="model SHA-256"):
        _run(server.url, tmp_path / "b.json", mmlu, arc, model_sha256="deadbeef")
    with pytest.raises(evaluator.EvaluationError, match=re.escape("llama.cpp commit")):
        _run(server.url, tmp_path / "c.json", mmlu, arc, llama_commit="a" * 40)
    with pytest.raises(evaluator.EvaluationError, match="server"):
        _run(server.url, tmp_path / "d.json", mmlu, arc, server="http://user:pass@127.0.0.1:8080")
    with pytest.raises(evaluator.EvaluationError, match="server"):
        _run(server.url, tmp_path / "e.json", mmlu, arc, server="ftp://127.0.0.1")
    assert server.requests == []


def test_cli_runs_and_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "cli.json"

    code = evaluator.main(
        [
            "--server",
            server.url,
            "--arm",
            "Q4_K_M",
            "--model-sha256",
            MODEL_SHA256,
            "--llama-commit",
            LLAMA_COMMIT,
            "--out",
            str(out),
        ],
        **_loaders(mmlu, arc),
    )

    assert code == 0
    published = json.loads(out.read_text(encoding="utf-8"))
    assert published["provenance"]["arm"] == "Q4_K_M"
    assert evaluator.artifact_paths(out)["completion"].is_file()


def test_cli_exits_nonzero_without_a_completion_marker_when_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmlu, arc = _mmlu_items(3), _arc_items(2)
    _configure(monkeypatch, mmlu, arc)
    out = tmp_path / "unreachable.json"

    with pytest.raises(SystemExit) as exit_info:
        evaluator.main(
            [
                "--server",
                "http://127.0.0.1:1",
                "--arm",
                "BF16",
                "--model-sha256",
                MODEL_SHA256,
                "--llama-commit",
                LLAMA_COMMIT,
                "--out",
                str(out),
            ],
            **_loaders(mmlu, arc),
        )

    assert exit_info.value.code == 2
    paths = evaluator.artifact_paths(out)
    assert not paths["completion"].exists()
    assert not paths["result"].exists()
    assert paths["failure"].is_file()
