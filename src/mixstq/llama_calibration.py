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
FORMAT_VERSION = 2
COMMIT_FORMAT = f"{FORMAT}.commit"
COMMIT_VERSION = 1
PUBLICATION_PROTOCOL = "staged-pair-with-atomic-commit-marker-v1"
CANONICAL_PER_DOMAIN = 32
CANONICAL_MIN_CHARS = 200
MAX_SEPARATOR_UTF8_BYTES = 16
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
    source_sha256: str
    normalized_sha256: str


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
                source_sha256=_sha256(raw.encode("utf-8")),
                normalized_sha256=_sha256(text.encode("utf-8")),
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
    for record_separator_count in range(1, MAX_SEPARATOR_UTF8_BYTES + 1):
        separator = "\n\n" + ("\x1e" * record_separator_count) + "\n\n"
        if len(separator.encode("utf-8")) > MAX_SEPARATOR_UTF8_BYTES:
            break
        if all(separator not in text for text in texts):
            return separator
    raise RuntimeError(
        "no collision-free record separator exists within the 16-byte UTF-8 limit"
    )


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


def commit_marker_path(manifest_path: Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.with_name(f"{manifest_path.name}.commit")


def _fsync_directories(paths: Sequence[Path]) -> None:
    parents = dict.fromkeys(path.parent for path in paths)
    for parent in parents:
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _commit_marker_bytes(corpus_bytes: bytes, manifest_bytes: bytes) -> bytes:
    marker = {
        "format": COMMIT_FORMAT,
        "version": COMMIT_VERSION,
        "protocol": PUBLICATION_PROTOCOL,
        "corpus_sha256": _sha256(corpus_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
    }
    return (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_pair(
    out: Path,
    corpus_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    marker_path = commit_marker_path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    published: list[Path] = []
    try:
        staged.append(_stage_bytes(out, corpus_bytes))
        staged.append(_stage_bytes(manifest_path, manifest_bytes))
        staged.append(
            _stage_bytes(marker_path, _commit_marker_bytes(corpus_bytes, manifest_bytes))
        )
        os.link(staged[0], out)
        published.append(out)
        os.link(staged[1], manifest_path)
        published.append(manifest_path)
        _fsync_directories((out, manifest_path))
        os.link(staged[2], marker_path)
        published.append(marker_path)
        _fsync_directories((marker_path,))
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


def require_committed_corpus(out: Path, manifest_path: Path) -> dict[str, Any]:
    out = Path(out)
    manifest_path = Path(manifest_path)
    marker_path = commit_marker_path(manifest_path)
    missing = [path for path in (out, manifest_path, marker_path) if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"calibration publication is not committed; missing: {names}")

    corpus_bytes = out.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    marker_bytes = marker_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
        marker = json.loads(marker_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("committed calibration metadata is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or not isinstance(marker, dict):
        raise RuntimeError("committed calibration metadata must be JSON objects")

    expected_marker = {
        "format": COMMIT_FORMAT,
        "version": COMMIT_VERSION,
        "protocol": PUBLICATION_PROTOCOL,
        "corpus_sha256": _sha256(corpus_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
    }
    if marker != expected_marker:
        raise RuntimeError("calibration commit marker does not match the published pair")
    corpus = manifest.get("corpus")
    publication = manifest.get("publication")
    if not isinstance(corpus, dict) or corpus.get("sha256") != expected_marker["corpus_sha256"]:
        raise RuntimeError("manifest corpus identity does not match the committed corpus")
    if (
        not isinstance(publication, dict)
        or publication.get("protocol") != PUBLICATION_PROTOCOL
    ):
        raise RuntimeError("manifest does not require the committed publication protocol")
    return manifest


def _validate_publication_paths(out: Path, manifest_path: Path) -> Path:
    marker_path = commit_marker_path(manifest_path)
    resolved = [path.resolve() for path in (out, manifest_path, marker_path)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output, manifest, and commit marker paths must be different")
    existing = [path for path in (out, manifest_path, marker_path) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        if marker_path not in existing:
            raise FileExistsError(
                f"uncommitted calibration publication blocks no-overwrite build: {names}"
            )
        try:
            require_committed_corpus(out, manifest_path)
        except RuntimeError as error:
            raise FileExistsError(
                f"invalid committed calibration publication blocks no-overwrite build: {names}"
            ) from error
        raise FileExistsError(f"committed calibration publication already exists: {names}")
    return marker_path


def _build_corpus_with_loader(
    out: Path,
    manifest_path: Path,
    *,
    per_domain: int,
    min_chars: int,
    load_dataset_fn: LoadDataset,
) -> dict[str, Any]:
    out = Path(out)
    manifest_path = Path(manifest_path)
    if per_domain < 1:
        raise ValueError("per_domain must be at least 1")
    if min_chars < 1:
        raise ValueError("min_chars must be at least 1")
    _validate_publication_paths(out, manifest_path)

    records = [
        record
        for spec in DATASETS
        for record in _select_records(spec, per_domain, min_chars, load_dataset_fn)
    ]
    texts = [record.text for record in records]
    ordered_normalized_sha256 = [record.normalized_sha256 for record in records]
    aggregate_hash = _ordered_hash(ordered_normalized_sha256)
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
                "source_sha256": record.source_sha256,
                "normalized_sha256": record.normalized_sha256,
            }
            for ordinal, record in enumerate(records)
        ],
        "ordered_normalized_sha256": ordered_normalized_sha256,
        "aggregate_ordered_normalized_sha256": aggregate_hash,
        "hash_semantics": {
            "source_sha256": (
                "SHA-256 over the raw UTF-8 source field before normalization; provenance only."
            ),
            "normalized_sha256": (
                "SHA-256 over emitted normalized UTF-8 record text; this feeds the ordered "
                "aggregate identity."
            ),
            "aggregate_ordered_normalized_sha256": (
                "SHA-256 over the ASCII concatenation of normalized_sha256 values in corpus "
                "order."
            ),
            "corpus_sha256": (
                "SHA-256 over the complete emitted corpus bytes, including separators; source "
                "hashes do not feed corpus identity."
            ),
        },
        "corpus": {
            "encoding": "UTF-8",
            "separator": separator,
            "byte_size": len(corpus_bytes),
            "sha256": _sha256(corpus_bytes),
        },
        "publication": {
            "protocol": PUBLICATION_PROTOCOL,
            "commit_marker": "manifest path with '.commit' appended",
            "consumer_requirement": (
                "Consumers must call require_committed_corpus and reject missing or mismatched "
                "commit state."
            ),
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
    return require_committed_corpus(out, manifest_path)


def build_corpus(
    out: Path,
    manifest_path: Path,
    *,
    per_domain: int = CANONICAL_PER_DOMAIN,
    min_chars: int = CANONICAL_MIN_CHARS,
) -> dict[str, Any]:
    if per_domain != CANONICAL_PER_DOMAIN:
        raise ValueError(f"per_domain must be exactly {CANONICAL_PER_DOMAIN}")
    if min_chars != CANONICAL_MIN_CHARS:
        raise ValueError(f"min_chars must be exactly {CANONICAL_MIN_CHARS}")
    out = Path(out)
    manifest_path = Path(manifest_path)
    _validate_publication_paths(out, manifest_path)
    return _build_corpus_with_loader(
        out,
        manifest_path,
        per_domain=per_domain,
        min_chars=min_chars,
        load_dataset_fn=_default_loader(),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pinned, deterministic plain-text llama.cpp imatrix corpus"
    )
    parser.add_argument("--out", required=True, type=Path, help="new UTF-8 corpus path")
    parser.add_argument("--manifest", required=True, type=Path, help="new JSON manifest path")
    parser.add_argument("--per-domain", type=int, default=CANONICAL_PER_DOMAIN)
    parser.add_argument("--min-chars", type=int, default=CANONICAL_MIN_CHARS)
    args = parser.parse_args(argv)
    if args.per_domain != CANONICAL_PER_DOMAIN:
        parser.error(f"--per-domain must be exactly {CANONICAL_PER_DOMAIN}")
    if args.min_chars != CANONICAL_MIN_CHARS:
        parser.error(f"--min-chars must be exactly {CANONICAL_MIN_CHARS}")
    return args


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    manifest = build_corpus(
        args.out,
        args.manifest,
        per_domain=args.per_domain,
        min_chars=args.min_chars,
    )
    committed = require_committed_corpus(args.out, args.manifest)
    print(
        json.dumps(
            {
                "corpus": str(args.out),
                "manifest": str(args.manifest),
                "commit_marker": str(commit_marker_path(args.manifest)),
                "committed": True,
                "records": len(manifest["records"]),
                "corpus_sha256": committed["corpus"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
