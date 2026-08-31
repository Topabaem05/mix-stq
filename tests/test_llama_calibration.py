from __future__ import annotations

import hashlib
import json
import os
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

    calibration.build_corpus(
        first_out,
        first_manifest,
        per_domain=2,
        min_chars=200,
        load_dataset_fn=_loader(rows, calls),
    )
    calibration.build_corpus(
        second_out,
        second_manifest,
        per_domain=2,
        min_chars=200,
        load_dataset_fn=_loader(rows, calls),
    )

    assert first_out.read_bytes() == second_out.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
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
        "wiki",
        "code",
        "code",
        "chat",
        "chat",
    ]

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
    expected_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in normalized_records
    ]
    assert manifest["ordered_record_sha256"] == expected_hashes
    assert [record["sha256"] for record in manifest["records"]] == expected_hashes
    assert manifest["aggregate_ordered_sha256"] == hashlib.sha256(
        "".join(expected_hashes).encode("ascii")
    ).hexdigest()
    assert manifest["corpus"]["byte_size"] == len(corpus_bytes)
    assert manifest["corpus"]["sha256"] == hashlib.sha256(corpus_bytes).hexdigest()
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


def test_normalization_is_canonical_and_preserves_code_layout() -> None:
    raw = "  Cafe\u0301  \r\n    return 1\t \r\n\r\n"
    assert calibration.normalize_text(raw) == "Café\n    return 1"
    assert calibration.normalize_text(" \r\n\t ") == ""


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

    manifest = calibration.build_corpus(
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
        calibration.build_corpus(
            out,
            manifest,
            per_domain=2,
            min_chars=200,
            load_dataset_fn=_loader(rows, []),
        )

    assert not out.exists()
    assert not manifest.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", ["out", "manifest"])
def test_existing_artifact_is_preserved_before_loading(tmp_path: Path, existing: str) -> None:
    out = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    protected = out if existing == "out" else manifest
    protected.write_bytes(b"do-not-overwrite")
    calls = []

    with pytest.raises(FileExistsError, match=str(protected)):
        calibration.build_corpus(
            out,
            manifest,
            load_dataset_fn=_loader(_rows_by_domain(32), calls),
        )

    assert protected.read_bytes() == b"do-not-overwrite"
    assert not (manifest if existing == "out" else out).exists()
    assert calls == []


def test_second_atomic_publish_failure_rolls_back_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    real_link = os.link
    link_calls = 0

    def fail_second_link(source, destination):
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("simulated manifest publish failure")
        return real_link(source, destination)

    monkeypatch.setattr(calibration.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated manifest publish failure"):
        calibration.build_corpus(
            out,
            manifest,
            per_domain=1,
            min_chars=200,
            load_dataset_fn=_loader(_rows_by_domain(1), []),
        )

    assert not out.exists()
    assert not manifest.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_requires_paths_and_exposes_selection_defaults() -> None:
    args = calibration.parse_args(["--out", "corpus.txt", "--manifest", "manifest.json"])
    assert args.out == Path("corpus.txt")
    assert args.manifest == Path("manifest.json")
    assert args.per_domain == 32
    assert args.min_chars == 200


def test_default_build_records_exactly_96_ordered_hashes(tmp_path: Path) -> None:
    manifest = calibration.build_corpus(
        tmp_path / "corpus.txt",
        tmp_path / "manifest.json",
        load_dataset_fn=_loader(_rows_by_domain(32), []),
    )

    assert len(manifest["records"]) == 96
    assert len(manifest["ordered_record_sha256"]) == 96
