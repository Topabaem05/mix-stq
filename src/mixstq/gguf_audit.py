"""Streaming physical-bpw and provenance audit for the pinned Qwen3.8-27B GGUF arms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import numbers
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

EXPECTED_ARCHITECTURE = "qwen35"
ARCHITECTURE_KEY = "general.architecture"
FILE_TYPE_KEY = "general.file_type"
SHA256_CHUNK_BYTES = 1024 * 1024
SUMMARY_FIELDS = (
    "sha256",
    "file_bytes",
    "architecture",
    "file_type",
    "tensor_count",
    "tensor_elements",
    "tensor_payload_bytes",
    "physical_bpw",
    "payload_bpw",
    "tensors_by_type",
)
READER_ERRORS = (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError)


class AuditError(ValueError):
    pass


@dataclass(frozen=True)
class TensorRecord:
    """One normalized GGUF tensor: identity, quantization type, and its data span."""

    name: str
    tensor_type: str
    elements: int
    nbytes: int
    offset: int


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise AuditError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise AuditError(f"{label} must be an integer of at least {minimum}")
    return normalized


def _text(value: object, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditError(f"{label} is not valid UTF-8") from error
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{label} must be a nonempty string")
    return value.strip()


def _resolve_architecture(metadata: Mapping[str, object], expected_arch: str) -> str:
    expected = _text(expected_arch, "expected GGUF architecture").casefold()
    if ARCHITECTURE_KEY not in metadata:
        raise AuditError(f"GGUF metadata is missing {ARCHITECTURE_KEY}")
    found = _text(metadata[ARCHITECTURE_KEY], f"GGUF metadata {ARCHITECTURE_KEY}").casefold()
    if found != expected:
        raise AuditError(f"GGUF architecture must be {expected}, found {found}")
    return found


def _resolve_file_type(metadata: Mapping[str, object]) -> int | str:
    if FILE_TYPE_KEY not in metadata:
        raise AuditError(f"GGUF metadata is missing {FILE_TYPE_KEY}")
    value = metadata[FILE_TYPE_KEY]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return _text(value, f"GGUF metadata {FILE_TYPE_KEY}")
    return int(value)


def _validated_records(tensors: Sequence[TensorRecord]) -> list[TensorRecord]:
    if isinstance(tensors, (str, bytes, Mapping)) or not isinstance(tensors, Sequence):
        raise AuditError("GGUF tensor records must be a sequence")
    records: list[TensorRecord] = []
    for tensor in tensors:
        if not isinstance(tensor, TensorRecord):
            raise AuditError("GGUF tensor records must be TensorRecord instances")
        records.append(
            TensorRecord(
                name=_text(tensor.name, "GGUF tensor name"),
                tensor_type=_text(tensor.tensor_type, "GGUF tensor type"),
                elements=_integer(tensor.elements, "GGUF tensor elements", minimum=1),
                nbytes=_integer(tensor.nbytes, "GGUF tensor payload bytes", minimum=1),
                offset=_integer(tensor.offset, "GGUF tensor data offset", minimum=0),
            )
        )
    names = {record.name for record in records}
    if len(names) != len(records):
        raise AuditError("GGUF tensor inventory contains a duplicate tensor name")
    ordered = sorted(records, key=lambda record: record.offset)
    for previous, current in itertools.pairwise(ordered):
        if current.offset < previous.offset + previous.nbytes:
            raise AuditError("GGUF tensor data spans overlap")
    return records


def _tensors_by_type(records: Sequence[TensorRecord]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = grouped.setdefault(
            record.tensor_type, {"count": 0, "elements": 0, "bytes": 0}
        )
        bucket["count"] += 1
        bucket["elements"] += record.elements
        bucket["bytes"] += record.nbytes
    return dict(sorted(grouped.items()))


def _finite_bpw(numerator: int, elements: int, label: str) -> float:
    value = numerator * 8.0 / elements
    if not math.isfinite(value) or value <= 0:
        raise AuditError(f"{label} is not a finite positive number")
    return value


def summarize_records(
    file_bytes: int,
    metadata: Mapping[str, object],
    tensors: Sequence[TensorRecord],
    *,
    expected_arch: str = EXPECTED_ARCHITECTURE,
) -> dict[str, object]:
    """Compute the audited GGUF summary from already-normalized tensor records."""

    if not isinstance(metadata, Mapping):
        raise AuditError("GGUF metadata must be a mapping")
    file_bytes = _integer(file_bytes, "GGUF file bytes", minimum=1)
    architecture = _resolve_architecture(metadata, expected_arch)
    file_type = _resolve_file_type(metadata)
    records = _validated_records(tensors)
    tensor_elements = sum(record.elements for record in records)
    tensor_payload_bytes = sum(record.nbytes for record in records)
    if tensor_elements <= 0:
        raise AuditError("GGUF has no tensor elements")
    if tensor_payload_bytes > file_bytes:
        raise AuditError("GGUF tensor payload bytes exceed the file size")
    return {
        "file_bytes": file_bytes,
        "architecture": architecture,
        "file_type": file_type,
        "tensor_count": len(records),
        "tensor_elements": tensor_elements,
        "tensor_payload_bytes": tensor_payload_bytes,
        "physical_bpw": _finite_bpw(file_bytes, tensor_elements, "GGUF physical bpw"),
        "payload_bpw": _finite_bpw(tensor_payload_bytes, tensor_elements, "GGUF payload bpw"),
        "tensors_by_type": _tensors_by_type(records),
    }


def load_reader_class() -> type:
    """Import GGUFReader from the pinned llama.cpp gguf package only when it is needed."""

    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise AuditError("the pinned gguf package is required to read GGUF files") from error
    return GGUFReader


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(SHA256_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as error:
        raise AuditError(f"GGUF model could not be read: {path}") from error
    return digest.hexdigest()


def _field_value(fields: Mapping[str, object], key: str) -> object | None:
    field = fields.get(key)
    if field is None:
        return None
    contents = getattr(field, "contents", None)
    if not callable(contents):
        raise AuditError("GGUF reader exposes an unsupported metadata field surface")
    try:
        return contents()
    except READER_ERRORS as error:
        raise AuditError(f"GGUF reader could not resolve metadata {key}") from error


def _reader_metadata(reader: object) -> dict[str, object]:
    fields = getattr(reader, "fields", None)
    if not isinstance(fields, Mapping):
        raise AuditError("GGUF reader does not expose a metadata field mapping")
    metadata: dict[str, object] = {}
    for key in (ARCHITECTURE_KEY, FILE_TYPE_KEY):
        value = _field_value(fields, key)
        if value is not None:
            metadata[key] = value
    return metadata


def _type_name(value: object) -> object:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else value


def _reader_tensors(reader: object) -> list[TensorRecord]:
    tensors = getattr(reader, "tensors", None)
    if isinstance(tensors, (str, bytes, Mapping)) or not isinstance(tensors, Sequence):
        raise AuditError("GGUF reader does not expose a tensor sequence")
    records = []
    for tensor in tensors:
        # Tensor payloads are never touched here; only shape/type/span metadata is read.
        try:
            record = TensorRecord(
                name=tensor.name,
                tensor_type=_type_name(tensor.tensor_type),
                elements=tensor.n_elements,
                nbytes=tensor.n_bytes,
                offset=tensor.data_offset,
            )
        except READER_ERRORS as error:
            raise AuditError("GGUF reader exposes an unsupported tensor record") from error
        records.append(record)
    return records


def summarize_gguf(
    path: Path,
    expected_arch: str = EXPECTED_ARCHITECTURE,
    *,
    reader_factory=None,
) -> dict[str, object]:
    """Audit one GGUF file by streaming its SHA-256 and reading only tensor metadata."""

    path = Path(path)
    if not path.is_file():
        raise AuditError(f"GGUF model is not a readable file: {path}")
    file_bytes = path.stat().st_size
    sha256 = _sha256_stream(path)
    if reader_factory is None:

        def reader_factory(model: Path) -> object:
            return load_reader_class()(os.fspath(model))

    try:
        reader = reader_factory(path)
    except AuditError:
        raise
    except READER_ERRORS as error:
        raise AuditError(f"GGUF reader could not open the model: {path}") from error
    summary = {
        "sha256": sha256,
        **summarize_records(
            file_bytes,
            _reader_metadata(reader),
            _reader_tensors(reader),
            expected_arch=expected_arch,
        ),
    }
    return {field: summary[field] for field in SUMMARY_FIELDS}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_audit(path: Path, summary: Mapping[str, object]) -> None:
    """Publish the audit JSON atomically and never overwrite an existing audit."""

    path = Path(path)
    data = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except OSError as error:
        raise AuditError(f"audit output directory is not writable: {path.parent}") from error
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditError(f"audit output already exists: {path}") from error
        except OSError as error:
            raise AuditError(f"audit output could not be published: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-arch", default=EXPECTED_ARCHITECTURE)
    return parser


def main(argv: Sequence[str] | None = None, *, reader_factory=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = summarize_gguf(args.model, args.expected_arch, reader_factory=reader_factory)
        publish_audit(args.out, summary)
    except AuditError as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
