from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path

import gguf_audit
import pytest

ARCHITECTURE = gguf_audit.EXPECTED_ARCHITECTURE
FILE_TYPE = 30


def _metadata(**overrides: object) -> dict[str, object]:
    metadata = {"general.architecture": ARCHITECTURE, "general.file_type": FILE_TYPE}
    metadata.update(overrides)
    return metadata


def _records() -> list[gguf_audit.TensorRecord]:
    return [
        gguf_audit.TensorRecord(
            name="blk.0.attn_q.weight",
            tensor_type="Q4_K",
            elements=600,
            nbytes=240,
            offset=0,
        ),
        gguf_audit.TensorRecord(
            name="blk.0.ffn_down.weight",
            tensor_type="IQ3_XXS",
            elements=400,
            nbytes=160,
            offset=240,
        ),
    ]


class _FakeField:
    def __init__(self, value: object) -> None:
        self._value = value

    def contents(self) -> object:
        return self._value


class _FakeTensor:
    def __init__(self, record: gguf_audit.TensorRecord) -> None:
        self.name = record.name
        self.tensor_type = types.SimpleNamespace(name=record.tensor_type)
        self.n_elements = record.elements
        self.n_bytes = record.nbytes
        self.data_offset = record.offset

    @property
    def data(self) -> object:
        raise AssertionError("the auditor must never materialize tensor payloads")


class _FakeReader:
    def __init__(self, metadata: dict[str, object], records: list[gguf_audit.TensorRecord]) -> None:
        self.fields = {key: _FakeField(value) for key, value in metadata.items()}
        self.tensors = [_FakeTensor(record) for record in records]


def _reader_factory(metadata: dict[str, object], records: list[gguf_audit.TensorRecord]):
    def factory(path: Path) -> _FakeReader:
        assert Path(path).is_file()
        return _FakeReader(metadata, records)

    return factory


def _write_model(tmp_path: Path, size: int = 500) -> Path:
    model = tmp_path / "qwen38-27b-iq3-xxs.gguf"
    model.write_bytes(bytes(range(256)) * (size // 256) + bytes(size % 256))
    assert model.stat().st_size == size
    return model


def test_summarize_records_reports_physical_and_payload_bpw() -> None:
    summary = gguf_audit.summarize_records(500, _metadata(), _records())

    assert summary["file_bytes"] == 500
    assert summary["tensor_count"] == 2
    assert summary["tensor_elements"] == 1000
    assert summary["tensor_payload_bytes"] == 400
    assert summary["physical_bpw"] == 4.0
    assert summary["payload_bpw"] == 3.2
    assert summary["architecture"] == ARCHITECTURE
    assert summary["file_type"] == FILE_TYPE
    assert summary["tensors_by_type"] == {
        "IQ3_XXS": {"count": 1, "elements": 400, "bytes": 160},
        "Q4_K": {"count": 1, "elements": 600, "bytes": 240},
    }


def test_summarize_records_rejects_wrong_architecture() -> None:
    with pytest.raises(gguf_audit.AuditError, match="architecture"):
        gguf_audit.summarize_records(500, _metadata(**{"general.architecture": "llama"}), _records())


def test_summarize_records_accepts_case_insensitive_architecture() -> None:
    summary = gguf_audit.summarize_records(
        500, _metadata(**{"general.architecture": ARCHITECTURE.upper()}), _records()
    )

    assert summary["architecture"] == ARCHITECTURE


def test_summarize_records_rejects_missing_file_type() -> None:
    metadata = {"general.architecture": ARCHITECTURE}
    with pytest.raises(gguf_audit.AuditError, match=re.escape("general.file_type")):
        gguf_audit.summarize_records(500, metadata, _records())


def test_summarize_records_rejects_zero_elements() -> None:
    with pytest.raises(gguf_audit.AuditError, match="no tensor elements"):
        gguf_audit.summarize_records(500, _metadata(), [])

    zero = [
        gguf_audit.TensorRecord(
            name="blk.0.attn_q.weight", tensor_type="Q4_K", elements=0, nbytes=240, offset=0
        )
    ]
    with pytest.raises(gguf_audit.AuditError, match="elements"):
        gguf_audit.summarize_records(500, _metadata(), zero)


def test_summarize_records_rejects_overlapping_tensor_spans() -> None:
    records = _records()
    overlapping = [
        records[0],
        gguf_audit.TensorRecord(
            name=records[1].name,
            tensor_type=records[1].tensor_type,
            elements=records[1].elements,
            nbytes=records[1].nbytes,
            offset=239,
        ),
    ]
    with pytest.raises(gguf_audit.AuditError, match="overlap"):
        gguf_audit.summarize_records(500, _metadata(), overlapping)


def test_summarize_records_rejects_nonfinite_bpw() -> None:
    single = [
        gguf_audit.TensorRecord(
            name="blk.0.attn_q.weight", tensor_type="Q4_K", elements=1, nbytes=1, offset=0
        )
    ]
    with pytest.raises(gguf_audit.AuditError, match="finite"):
        gguf_audit.summarize_records(10**308, _metadata(), single)


def test_summarize_records_rejects_payload_larger_than_file() -> None:
    with pytest.raises(gguf_audit.AuditError, match="payload"):
        gguf_audit.summarize_records(399, _metadata(), _records())


def test_summarize_records_rejects_duplicate_tensor_names() -> None:
    records = _records()
    duplicated = [
        records[0],
        gguf_audit.TensorRecord(
            name=records[0].name,
            tensor_type=records[1].tensor_type,
            elements=records[1].elements,
            nbytes=records[1].nbytes,
            offset=records[1].offset,
        ),
    ]
    with pytest.raises(gguf_audit.AuditError, match="duplicate"):
        gguf_audit.summarize_records(500, _metadata(), duplicated)


def test_summarize_records_rejects_invalid_numbers() -> None:
    with pytest.raises(gguf_audit.AuditError, match="file bytes"):
        gguf_audit.summarize_records(0, _metadata(), _records())
    with pytest.raises(gguf_audit.AuditError, match="file bytes"):
        gguf_audit.summarize_records(500.0, _metadata(), _records())
    negative = [
        gguf_audit.TensorRecord(
            name="blk.0.attn_q.weight", tensor_type="Q4_K", elements=600, nbytes=240, offset=-1
        )
    ]
    with pytest.raises(gguf_audit.AuditError, match="offset"):
        gguf_audit.summarize_records(500, _metadata(), negative)


def test_summarize_gguf_streams_sha256_and_normalizes_the_reader(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    summary = gguf_audit.summarize_gguf(
        model, reader_factory=_reader_factory(_metadata(), _records())
    )

    assert set(summary) == set(gguf_audit.SUMMARY_FIELDS)
    assert summary["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert summary["file_bytes"] == 500
    assert summary["physical_bpw"] == 4.0
    assert summary["payload_bpw"] == 3.2
    assert summary["architecture"] == ARCHITECTURE


def test_summarize_gguf_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(gguf_audit.AuditError, match="not a readable file"):
        gguf_audit.summarize_gguf(
            tmp_path / "absent.gguf", reader_factory=_reader_factory(_metadata(), _records())
        )


def test_summarize_gguf_propagates_architecture_mismatch(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    factory = _reader_factory(_metadata(**{"general.architecture": "llama"}), _records())
    with pytest.raises(gguf_audit.AuditError, match="architecture"):
        gguf_audit.summarize_gguf(model, reader_factory=factory)


def test_summarize_gguf_rejects_an_unusable_reader(tmp_path: Path) -> None:
    model = _write_model(tmp_path)

    def factory(_path: Path) -> object:
        return object()

    with pytest.raises(gguf_audit.AuditError, match="reader"):
        gguf_audit.summarize_gguf(model, reader_factory=factory)


def test_reader_class_is_imported_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("gguf")
    fake.GGUFReader = object
    monkeypatch.setitem(sys.modules, "gguf", fake)
    assert gguf_audit.load_reader_class() is object

    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(gguf_audit.AuditError, match="gguf"):
        gguf_audit.load_reader_class()


def test_cli_publishes_the_audit_atomically_and_refuses_overwrite(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    out = tmp_path / "audit" / "iq3_xxs.json"
    factory = _reader_factory(_metadata(), _records())

    assert gguf_audit.main(["--model", str(model), "--out", str(out)], reader_factory=factory) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["physical_bpw"] == 4.0
    assert payload["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert set(payload) == set(gguf_audit.SUMMARY_FIELDS)
    assert not list(out.parent.glob(".*.tmp"))

    with pytest.raises(SystemExit) as exit_info:
        gguf_audit.main(["--model", str(model), "--out", str(out)], reader_factory=factory)
    assert exit_info.value.code == 2
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_cli_rejects_a_missing_model_without_writing_output(tmp_path: Path) -> None:
    out = tmp_path / "unused.json"
    factory = _reader_factory(_metadata(), _records())

    with pytest.raises(SystemExit) as exit_info:
        gguf_audit.main(
            ["--model", str(tmp_path / "absent.gguf"), "--out", str(out)], reader_factory=factory
        )

    assert exit_info.value.code == 2
    assert not out.exists()
    assert not list(tmp_path.iterdir())
