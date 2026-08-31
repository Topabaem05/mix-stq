from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "mixstq.llama-imatrix-calibration"
FORMAT_VERSION = 1
DOMAIN_ORDER = ("wiki", "code", "chat")


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    dataset_id: str
    config: str | None
    split: str
    revision: str
    field: str

    def manifest_entry(self) -> dict[str, str | None]:
        return {
            "domain": self.domain,
            "id": self.dataset_id,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
            "field": self.field,
        }


DATASETS = (
    DatasetSpec(
        domain="wiki",
        dataset_id="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        split="train",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        field="text",
    ),
    DatasetSpec(
        domain="code",
        dataset_id="codeparrot/codeparrot-clean-valid",
        config=None,
        split="train",
        revision="4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
        field="content",
    ),
    DatasetSpec(
        domain="chat",
        dataset_id="HuggingFaceH4/ultrachat_200k",
        config=None,
        split="train_sft",
        revision="8049631c405ae6576f93f445c6b8166f76f5505a",
        field="prompt",
    ),
)


@dataclass(frozen=True)
class SelectedRecord:
    domain: str
    domain_index: int
    source_index: int
    text: str
    sha256: str


LoadDataset = Callable[..., Iterable[Mapping[str, object]]]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _select_records(
    spec: DatasetSpec,
    per_domain: int,
    min_chars: int,
    load_dataset_fn: LoadDataset,
) -> list[SelectedRecord]:
    stream = load_dataset_fn(
        spec.dataset_id,
        spec.config,
        split=spec.split,
        revision=spec.revision,
        streaming=True,
    )
    selected: list[SelectedRecord] = []
    for source_index, row in enumerate(stream):
        raw = row.get(spec.field)
        if not isinstance(raw, str):
            continue
        text = normalize_text(raw)
        if len(text) < min_chars:
            continue
        selected.append(
            SelectedRecord(
                domain=spec.domain,
                domain_index=len(selected),
                source_index=source_index,
                text=text,
                sha256=_sha256(text.encode("utf-8")),
            )
        )
        if len(selected) == per_domain:
            break
    if len(selected) != per_domain:
        raise RuntimeError(
            f"{spec.domain} stream supplied {len(selected)} qualifying records; "
            f"required {per_domain}"
        )
    return selected


def _ordered_hash(record_hashes: Sequence[str]) -> str:
    return _sha256("".join(record_hashes).encode("ascii"))


def _choose_separator(texts: Sequence[str]) -> str:
    record_separator_count = 1
    while True:
        separator = "\n\n" + ("\x1e" * record_separator_count) + "\n\n"
        if all(separator not in text for text in texts):
            return separator
        record_separator_count += 1


def _stage_bytes(destination: Path, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
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


def _publish_pair(
    out: Path,
    corpus_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    published: list[Path] = []
    try:
        staged.append(_stage_bytes(out, corpus_bytes))
        staged.append(_stage_bytes(manifest_path, manifest_bytes))
        os.link(staged[0], out)
        published.append(out)
        os.link(staged[1], manifest_path)
        published.append(manifest_path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def _default_loader() -> LoadDataset:
    from datasets import load_dataset

    return load_dataset


def build_corpus(
    out: Path,
    manifest_path: Path,
    *,
    per_domain: int = 32,
    min_chars: int = 200,
    load_dataset_fn: LoadDataset | None = None,
) -> dict[str, Any]:
    out = Path(out)
    manifest_path = Path(manifest_path)
    if per_domain < 1:
        raise ValueError("per_domain must be at least 1")
    if min_chars < 1:
        raise ValueError("min_chars must be at least 1")
    if out.resolve() == manifest_path.resolve():
        raise ValueError("output and manifest paths must be different")
    for path in (out, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    loader = load_dataset_fn if load_dataset_fn is not None else _default_loader()
    records = [
        record
        for spec in DATASETS
        for record in _select_records(spec, per_domain, min_chars, loader)
    ]
    texts = [record.text for record in records]
    ordered_record_sha256 = [record.sha256 for record in records]
    aggregate_hash = _ordered_hash(ordered_record_sha256)
    separator = _choose_separator(texts)
    corpus_bytes = separator.join(texts).encode("utf-8")
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "datasets": [spec.manifest_entry() for spec in DATASETS],
        "selection": {
            "domain_order": list(DOMAIN_ORDER),
            "per_domain": per_domain,
            "min_chars": min_chars,
            "length_unit": "Unicode code points after normalization",
            "rule": (
                "For each domain in domain_order, select the first per_domain source-order "
                "records whose normalized field is nonempty and at least min_chars long."
            ),
            "normalization": (
                "Unicode NFC; CRLF and CR converted to LF; trailing whitespace removed from "
                "each line; surrounding Unicode whitespace stripped."
            ),
        },
        "records": [
            {
                "ordinal": ordinal,
                "domain": record.domain,
                "domain_index": record.domain_index,
                "source_index": record.source_index,
                "normalized_chars": len(record.text),
                "utf8_bytes": len(record.text.encode("utf-8")),
                "sha256": record.sha256,
            }
            for ordinal, record in enumerate(records)
        ],
        "ordered_record_sha256": ordered_record_sha256,
        "aggregate_ordered_sha256": aggregate_hash,
        "corpus": {
            "encoding": "UTF-8",
            "separator": separator,
            "byte_size": len(corpus_bytes),
            "sha256": _sha256(corpus_bytes),
        },
        "contamination": {
            "evaluation_datasets_touched": False,
            "excluded_evaluation_datasets": ["MMLU", "ARC"],
            "statement": (
                "This builder loads only the three pinned calibration datasets listed above; "
                "MMLU and ARC evaluation data are never loaded or selected."
            ),
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_pair(out, corpus_bytes, manifest_path, manifest_bytes)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pinned, deterministic plain-text llama.cpp imatrix corpus"
    )
    parser.add_argument("--out", required=True, type=Path, help="new UTF-8 corpus path")
    parser.add_argument("--manifest", required=True, type=Path, help="new JSON manifest path")
    parser.add_argument("--per-domain", type=int, default=32)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args(argv)
    if args.per_domain < 1:
        parser.error("--per-domain must be at least 1")
    if args.min_chars < 1:
        parser.error("--min-chars must be at least 1")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> int:
    args = parse_args(argv)
    manifest = build_corpus(
        args.out,
        args.manifest,
        per_domain=args.per_domain,
        min_chars=args.min_chars,
        load_dataset_fn=load_dataset_fn,
    )
    print(
        json.dumps(
            {
                "corpus": str(args.out),
                "manifest": str(args.manifest),
                "records": len(manifest["records"]),
                "corpus_sha256": manifest["corpus"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
