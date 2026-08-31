from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import llama_calibration as calibration
import pytest


def _text(label: str, length: int = 240) -> str:
    return f"{label}:" + ("x" * (length - len(label) - 1))


def _rows_by_domain(per_domain: int = 2) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {}
    for spec in calibration.DATASETS:
        values: list[dict[str, object]] = [
            {spec.field: None},
            {spec.field: " short "},
        ]
        values.extend(
            {spec.field: f"  {domain_index}-Cafe\u0301\r\n{_text(spec.domain)}  "}
            for domain_index in range(per_domain)
        )
        rows[spec.domain] = values
    return rows


def _loader(rows_by_domain, calls):
    def load_dataset(dataset_id, config=None, **kwargs):
        spec = next(spec for spec in calibration.DATASETS if spec.dataset_id == dataset_id)
        calls.append((dataset_id, config, kwargs))
        return iter(rows_by_domain[spec.domain])

    return load_dataset


def test_deterministic_corpus_revisions_order_and_hashes(tmp_path: Path) -> None:
    rows = _rows_by_domain()
    calls: list[tuple[str, str | None, dict[str, object]]] = []
    first_out = tmp_path / "first.txt"
    first_manifest = tmp_path / "first.json"
    second_out = tmp_path / "second.txt"
    second_manifest = tmp_path / "second.json"

    calibration._build_corpus_with_loader(
        first_out,
        first_manifest,
        per_domain=2,
        min_chars=200,
        load_dataset_fn=_loader(rows, calls),
    )
    calibration._build_corpus_with_loader(
        second_out,
        second_manifest,
        per_domain=2,
        min_chars=200,
        load_dataset_fn=_loader(rows, calls),
    )

    assert first_out.read_bytes() == second_out.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert calibration.commit_marker_path(first_manifest).read_bytes() == (
        calibration.commit_marker_path(second_manifest).read_bytes()
    )
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert calibration.require_committed_corpus(first_out, first_manifest) == manifest
    assert manifest["format"] == calibration.FORMAT
    assert manifest["version"] == calibration.FORMAT_VERSION
    assert manifest["selection"]["domain_order"] == ["wiki", "code", "chat"]
    assert manifest["selection"]["per_domain"] == 2
    assert manifest["selection"]["min_chars"] == 200
    expected_datasets = [
        {
            "domain": "wiki",
            "id": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "field": "text",
        },
        {
            "domain": "code",
            "id": "codeparrot/codeparrot-clean-valid",
            "config": None,
            "split": "train",
            "revision": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
            "field": "content",
        },
        {
            "domain": "chat",
            "id": "HuggingFaceH4/ultrachat_200k",
            "config": None,
            "split": "train_sft",
            "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
            "field": "prompt",
        },
    ]
    assert manifest["datasets"] == expected_datasets
    assert [record["domain"] for record in manifest["records"]] == [
        "wiki",
        "code",
        "chat",
        "wiki",
        "code",
        "chat",
    ]
    assert [record["domain_index"] for record in manifest["records"]] == [
        0,
        0,
        0,
        1,
        1,
        1,
    ]
    assert manifest["selection"]["per_domain_counts"] == {
        domain: {"selected_records": 2, "emitted_records": 2}
        for domain in calibration.DOMAIN_ORDER
    }
    assert "wiki[i], code[i], chat[i]" in manifest["serialization"][
        "ordering_contract"
    ]
    assert manifest["serialization"]["max_record_utf8_bytes"] == 512

    expected_calls = [
        (
            dataset["id"],
            dataset["config"],
            {
                "split": dataset["split"],
                "revision": dataset["revision"],
                "streaming": True,
            },
        )
        for dataset in expected_datasets
    ]
    assert calls == expected_calls * 2
    assert not any("mmlu" in dataset_id.lower() or "arc" in dataset_id.lower()
                   for dataset_id, _config, _kwargs in calls)

    corpus_bytes = first_out.read_bytes()
    separator = manifest["corpus"]["separator"]
    assert separator == "\n\n\x1e\n\n"
    assert len(separator.encode("utf-8")) <= 16
    normalized_records = first_out.read_text(encoding="utf-8").split(separator)
    expected_normalized_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in normalized_records
    ]
    selected_raw_texts = [
        rows[spec.domain][2 + domain_index][spec.field]
        for domain_index in range(2)
        for spec in calibration.DATASETS
    ]
    assert all(isinstance(text, str) for text in selected_raw_texts)
    expected_source_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in selected_raw_texts
    ]
    assert manifest["ordered_full_normalized_text_sha256"] == expected_normalized_hashes
    assert manifest["ordered_emitted_text_sha256"] == expected_normalized_hashes
    assert [record["raw_source_sha256"] for record in manifest["records"]] == (
        expected_source_hashes
    )
    assert [
        record["full_normalized_text_sha256"] for record in manifest["records"]
    ] == expected_normalized_hashes
    assert [record["emitted_text_sha256"] for record in manifest["records"]] == (
        expected_normalized_hashes
    )
    assert expected_source_hashes != expected_normalized_hashes
    expected_aggregate = hashlib.sha256(
        "".join(expected_normalized_hashes).encode("ascii")
    ).hexdigest()
    assert manifest["aggregate_ordered_full_normalized_text_sha256"] == (
        expected_aggregate
    )
    assert manifest["aggregate_ordered_emitted_text_sha256"] == expected_aggregate
    assert "provenance only" in manifest["hash_semantics"]["raw_source_sha256"]
    assert "before emission truncation" in (
        manifest["hash_semantics"]["full_normalized_text_sha256"]
    )
    assert "actually emitted" in manifest["hash_semantics"]["emitted_text_sha256"]
    assert "source hashes do not feed corpus identity" in (
        manifest["hash_semantics"]["corpus_sha256"]
    )
    assert manifest["corpus"]["byte_size"] == len(corpus_bytes)
    assert manifest["corpus"]["sha256"] == hashlib.sha256(corpus_bytes).hexdigest()
    assert manifest["imatrix_capacity"] == {
        "chunks": 128,
        "tokens_per_chunk": 512,
        "total_token_capacity": 65_536,
        "corpus_utf8_byte_upper_bound": 65_536,
        "corpus_utf8_byte_count": len(corpus_bytes),
        "byte_upper_bound_check_passed": True,
        "exact_tokenizer_preflight_required": True,
        "scope_note": (
            "The deterministic UTF-8 byte upper-bound gate is conservative bookkeeping; "
            "it does not replace the later exact llama.cpp tokenizer preflight against "
            "the 128 * 512-token capacity."
        ),
    }
    marker = json.loads(
        calibration.commit_marker_path(first_manifest).read_text(encoding="utf-8")
    )
    assert marker == {
        "format": calibration.COMMIT_FORMAT,
        "version": calibration.COMMIT_VERSION,
        "protocol": calibration.PUBLICATION_PROTOCOL,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(first_manifest.read_bytes()).hexdigest(),
    }
    assert separator not in normalized_records
    assert manifest["contamination"]["evaluation_datasets_touched"] is False
    assert manifest["contamination"]["excluded_evaluation_datasets"] == ["MMLU", "ARC"]


def test_separator_collision_uses_short_variant_and_corpus_round_trips() -> None:
    expected_records = [
        "first\n\n\x1e\n\nrecord",
        "second record",
        "third record",
    ]
    separator = calibration._choose_separator(expected_records)
    corpus = separator.join(expected_records)

    assert separator == "\n\n\x1e\x1e\n\n"
    assert len(separator.encode("utf-8")) <= 16
    assert all(separator not in record for record in expected_records)
    assert corpus.split(separator) == expected_records


def test_separator_exhaustion_fails_closed_within_utf8_limit() -> None:
    candidates = [
        "\n\n" + ("\x1e" * count) + "\n\n"
        for count in range(1, calibration.MAX_SEPARATOR_UTF8_BYTES + 1)
        if len(("\n\n" + ("\x1e" * count) + "\n\n").encode("utf-8"))
        <= calibration.MAX_SEPARATOR_UTF8_BYTES
    ]
    assert candidates
    assert max(len(candidate.encode("utf-8")) for candidate in candidates) == 16

    with pytest.raises(RuntimeError, match=r"no collision-free.*16-byte"):
        calibration._choose_separator(["".join(candidates)])


def test_normalization_is_canonical_and_preserves_code_layout() -> None:
    raw = "  Cafe\u0301  \r\n    return 1\t \r\n\r\n"
    assert calibration.normalize_text(raw) == "Café\n    return 1"
    assert calibration.normalize_text(" \r\n\t ") == ""


def test_unicode_safe_truncation_and_unambiguous_hash_semantics(tmp_path: Path) -> None:
    raw = "  " + ("a" * 510) + "€tail  "
    full_normalized = calibration.normalize_text(raw)
    expected_emitted = "a" * 510
    rows = {
        spec.domain: [{spec.field: raw}]
        for spec in calibration.DATASETS
    }
    out = tmp_path / "unicode.txt"
    manifest = calibration._build_corpus_with_loader(
        out,
        tmp_path / "unicode.json",
        per_domain=1,
        min_chars=200,
        load_dataset_fn=_loader(rows, []),
    )

    emitted = out.read_text(encoding="utf-8").split(manifest["corpus"]["separator"])
    assert emitted == [expected_emitted] * 3
    for record in manifest["records"]:
        assert record["full_normalized_chars"] == len(full_normalized)
        assert record["full_normalized_utf8_bytes"] == len(
            full_normalized.encode("utf-8")
        )
        assert record["emitted_text_chars"] == len(expected_emitted)
        assert record["emitted_text_utf8_bytes"] == 510
        assert record["emitted_text_was_truncated"] is True
        assert record["raw_source_sha256"] == hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()
        assert record["full_normalized_text_sha256"] == hashlib.sha256(
            full_normalized.encode("utf-8")
        ).hexdigest()
        assert record["emitted_text_sha256"] == hashlib.sha256(
            expected_emitted.encode("utf-8")
        ).hexdigest()


def test_corpus_byte_budget_fails_closed_before_publication(tmp_path: Path) -> None:
    per_domain = 44
    rows = {
        spec.domain: [
            {spec.field: f"{domain_index:03d}:" + ("x" * 600)}
            for domain_index in range(per_domain)
        ]
        for spec in calibration.DATASETS
    }
    out = tmp_path / "over-budget.txt"
    manifest = tmp_path / "over-budget.json"

    with pytest.raises(RuntimeError, match=r"exceeding.*65_?536|exceeding.*65536"):
        calibration._build_corpus_with_loader(
            out,
            manifest,
            per_domain=per_domain,
            min_chars=200,
            load_dataset_fn=_loader(rows, []),
        )

    assert not out.exists()
    assert not manifest.exists()
    assert not calibration.commit_marker_path(manifest).exists()
    assert list(tmp_path.iterdir()) == []


def test_minimum_character_boundary_is_inclusive_and_source_ordered(tmp_path: Path) -> None:
    rows = {}
    for spec in calibration.DATASETS:
        rows[spec.domain] = [
            {spec.field: "a" * 199},
            {spec.field: "b" * 200},
            {spec.field: "c" * 201},
        ]
    out = tmp_path / "boundary.txt"
    manifest_path = tmp_path / "boundary.json"

    manifest = calibration._build_corpus_with_loader(
        out,
        manifest_path,
        per_domain=1,
        min_chars=200,
        load_dataset_fn=_loader(rows, []),
    )

    assert [record["source_index"] for record in manifest["records"]] == [1, 1, 1]
    assert out.read_text(encoding="utf-8").split(manifest["corpus"]["separator"]) == [
        "b" * 200,
        "b" * 200,
        "b" * 200,
    ]


def test_insufficient_stream_leaves_no_artifacts_or_temps(tmp_path: Path) -> None:
    rows = _rows_by_domain(per_domain=2)
    code_field = next(spec.field for spec in calibration.DATASETS if spec.domain == "code")
    rows["code"] = [{code_field: _text("only-one")}]
    out = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(RuntimeError, match=r"code.*1.*2"):
        calibration._build_corpus_with_loader(
            out,
            manifest,
            per_domain=2,
            min_chars=200,
            load_dataset_fn=_loader(rows, []),
        )

    assert not out.exists()
    assert not manifest.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", ["out", "manifest", "marker"])
def test_existing_artifact_is_preserved_before_loading(tmp_path: Path, existing: str) -> None:
    out = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    paths = {
        "out": out,
        "manifest": manifest,
        "marker": calibration.commit_marker_path(manifest),
    }
    protected = paths[existing]
    protected.write_bytes(b"do-not-overwrite")
    calls = []

    with pytest.raises(FileExistsError, match=str(protected)):
        calibration._build_corpus_with_loader(
            out,
            manifest,
            per_domain=32,
            min_chars=200,
            load_dataset_fn=_loader(_rows_by_domain(32), calls),
        )

    assert protected.read_bytes() == b"do-not-overwrite"
    assert all(not path.exists() for name, path in paths.items() if name != existing)
    assert calls == []


def test_commit_marker_publish_failure_rolls_back_pair_and_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    real_link = os.link
    link_calls = 0

    def fail_second_link(source, destination):
        nonlocal link_calls
        link_calls += 1
        if link_calls == 3:
            raise OSError("simulated commit marker publish failure")
        return real_link(source, destination)

    monkeypatch.setattr(calibration.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated commit marker publish failure"):
        calibration._build_corpus_with_loader(
            out,
            manifest,
            per_domain=1,
            min_chars=200,
            load_dataset_fn=_loader(_rows_by_domain(1), []),
        )

    assert not out.exists()
    assert not manifest.exists()
    assert not calibration.commit_marker_path(manifest).exists()
    assert list(tmp_path.iterdir()) == []


def test_abrupt_interruption_before_commit_is_rejected_without_loading(
    tmp_path: Path,
) -> None:
    out = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    source_root = Path(calibration.__file__).resolve().parents[1]
    script = r"""
import os
import sys
from pathlib import Path
import mixstq.llama_calibration as calibration

out = Path(sys.argv[1])
manifest = Path(sys.argv[2])
real_link = os.link
link_calls = 0

def interrupt_before_commit(source, destination):
    global link_calls
    link_calls += 1
    if link_calls == 3:
        os._exit(73)
    return real_link(source, destination)

calibration.os.link = interrupt_before_commit
calibration._publish_pair(out, b"corpus", manifest, b"{}\n")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(out), str(manifest)],
        env=environment,
        check=False,
    )

    assert completed.returncode == 73
    assert out.is_file()
    assert manifest.is_file()
    assert not calibration.commit_marker_path(manifest).exists()
    with pytest.raises(RuntimeError, match="not committed"):
        calibration.require_committed_corpus(out, manifest)

    calls = []
    with pytest.raises(FileExistsError, match="uncommitted"):
        calibration._build_corpus_with_loader(
            out,
            manifest,
            per_domain=1,
            min_chars=200,
            load_dataset_fn=_loader(_rows_by_domain(1), calls),
        )
    assert calls == []


def test_cli_requires_paths_and_exposes_selection_defaults() -> None:
    args = calibration.parse_args(["--out", "corpus.txt", "--manifest", "manifest.json"])
    assert args.out == Path("corpus.txt")
    assert args.manifest == Path("manifest.json")
    assert args.per_domain == 32
    assert args.min_chars == 200


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--per-domain", "1", "--per-domain must be exactly 32"),
        ("--min-chars", "199", "--min-chars must be exactly 200"),
    ],
)
def test_cli_rejects_noncanonical_selection(
    option: str, value: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        calibration.parse_args(
            ["--out", "corpus.txt", "--manifest", "manifest.json", option, value]
        )

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"per_domain": 1}, "per_domain must be exactly 32"),
        ({"min_chars": 199}, "min_chars must be exactly 200"),
    ],
)
def test_public_builder_rejects_noncanonical_selection_before_loading(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        calibration.build_corpus(
            tmp_path / "corpus.txt",
            tmp_path / "manifest.json",
            **kwargs,
        )

    assert list(tmp_path.iterdir()) == []


def test_public_builder_rejects_existing_artifact_before_default_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corpus.txt"
    out.write_bytes(b"preserve")
    loader_calls = 0

    def unexpected_default_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("default loader must not run")

    monkeypatch.setattr(calibration, "_default_loader", unexpected_default_loader)
    with pytest.raises(FileExistsError, match="uncommitted"):
        calibration.build_corpus(out, tmp_path / "manifest.json")

    assert loader_calls == 0
    assert out.read_bytes() == b"preserve"


def test_public_default_build_records_exactly_96_ordered_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        calibration,
        "_default_loader",
        lambda: _loader(_rows_by_domain(32), []),
    )
    manifest = calibration.build_corpus(
        tmp_path / "corpus.txt",
        tmp_path / "manifest.json",
    )

    assert len(manifest["records"]) == 96
    assert len(manifest["ordered_full_normalized_text_sha256"]) == 96
    assert len(manifest["ordered_emitted_text_sha256"]) == 96
    assert manifest["selection"]["per_domain"] == 32
    assert manifest["selection"]["min_chars"] == 200
    assert manifest["selection"]["per_domain_counts"] == {
        domain: {"selected_records": 32, "emitted_records": 32}
        for domain in calibration.DOMAIN_ORDER
    }
    assert [record["domain"] for record in manifest["records"][:6]] == [
        "wiki",
        "code",
        "chat",
        "wiki",
        "code",
        "chat",
    ]
    assert all(
        record["emitted_text_utf8_bytes"] <= calibration.MAX_RECORD_UTF8_BYTES
        for record in manifest["records"]
    )
    assert manifest["corpus"]["byte_size"] <= 65_536


def test_cli_exposes_and_requires_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        calibration,
        "_default_loader",
        lambda: _loader(_rows_by_domain(32), []),
    )
    out = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "manifest.json"

    assert (
        calibration.main(["--out", str(out), "--manifest", str(manifest_path)]) == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["committed"] is True
    assert payload["commit_marker"] == str(calibration.commit_marker_path(manifest_path))
    committed = calibration.require_committed_corpus(out, manifest_path)
    assert payload["corpus_sha256"] == committed["corpus"]["sha256"]


def test_consumer_rejects_tampered_committed_pair(tmp_path: Path) -> None:
    out = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "manifest.json"
    calibration._build_corpus_with_loader(
        out,
        manifest_path,
        per_domain=1,
        min_chars=200,
        load_dataset_fn=_loader(_rows_by_domain(1), []),
    )
    out.write_bytes(out.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="commit marker does not match"):
        calibration.require_committed_corpus(out, manifest_path)
