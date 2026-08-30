from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from ltc_format import (
    BLOCK_BYTES,
    CODEBOOK_BYTES,
    QK_BLOCK,
    decode_tensor,
    encode_tensor,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGML_TYPE_LTC1_0 = 40

TYPE_UINT32 = 4
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT8 = 0


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", TYPE_STRING) + _string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", TYPE_UINT32) + struct.pack("<I", value)


def _kv_uint8_array(key: str, payload: bytes) -> bytes:
    return (
        _string(key)
        + struct.pack("<I", TYPE_ARRAY)
        + struct.pack("<I", TYPE_UINT8)
        + struct.pack("<Q", len(payload))
        + payload
    )


def codebook_key(tensor_name: str) -> str:
    return "ltc.codebook." + tensor_name


def write_gguf(path: Path, tensors: dict[str, dict[str, object]], alignment: int = 32) -> dict:
    metadata = [
        _kv_string("general.architecture", "mixstq-ltc"),
        _kv_uint32("ltc.block_size", QK_BLOCK),
        _kv_uint32("ltc.block_bytes", BLOCK_BYTES),
        _kv_uint32("ltc.codebook_bytes", CODEBOOK_BYTES),
        _kv_uint32("general.alignment", alignment),
    ]
    for name, entry in tensors.items():
        metadata.append(_kv_uint8_array(codebook_key(name), entry["codebook"]))

    info = []
    offset = 0
    for name, entry in tensors.items():
        shape = entry["shape"]
        payload = entry["payload"]
        info.append(
            _string(name)
            + struct.pack("<I", len(shape))
            + b"".join(struct.pack("<Q", int(d)) for d in shape)
            + struct.pack("<I", GGML_TYPE_LTC1_0)
            + struct.pack("<Q", offset)
        )
        padded = (len(payload) + alignment - 1) // alignment * alignment
        offset += padded

    header = (
        GGUF_MAGIC
        + struct.pack("<I", GGUF_VERSION)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(metadata))
        + b"".join(metadata)
        + b"".join(info)
    )
    pad = (-len(header)) % alignment
    blob = [header, b"\x00" * pad]
    for entry in tensors.values():
        payload = entry["payload"]
        blob.append(payload)
        blob.append(b"\x00" * ((-len(payload)) % alignment))
    data = b"".join(blob)
    path.write_bytes(data)
    return {"bytes": len(data), "tensors": len(tensors), "metadata_entries": len(metadata)}


def read_gguf(path: Path) -> dict:
    raw = path.read_bytes()
    cursor = 0

    def take(count: int) -> bytes:
        nonlocal cursor
        chunk = raw[cursor : cursor + count]
        cursor += count
        return chunk

    if take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = struct.unpack("<I", take(4))[0]
    tensor_count = struct.unpack("<Q", take(8))[0]
    kv_count = struct.unpack("<Q", take(8))[0]

    def read_string() -> str:
        length = struct.unpack("<Q", take(8))[0]
        return take(length).decode("utf-8")

    kv: dict[str, object] = {}
    for _ in range(kv_count):
        key = read_string()
        kind = struct.unpack("<I", take(4))[0]
        if kind == TYPE_STRING:
            kv[key] = read_string()
        elif kind == TYPE_UINT32:
            kv[key] = struct.unpack("<I", take(4))[0]
        elif kind == TYPE_ARRAY:
            element = struct.unpack("<I", take(4))[0]
            count = struct.unpack("<Q", take(8))[0]
            if element != TYPE_UINT8:
                raise ValueError("only uint8 arrays are supported")
            kv[key] = take(count)
        else:
            raise ValueError("unsupported metadata type %d" % kind)

    tensors = {}
    for _ in range(tensor_count):
        name = read_string()
        dims = struct.unpack("<I", take(4))[0]
        shape = [struct.unpack("<Q", take(8))[0] for _ in range(dims)]
        ggml_type = struct.unpack("<I", take(4))[0]
        offset = struct.unpack("<Q", take(8))[0]
        tensors[name] = {"shape": shape, "type": ggml_type, "offset": offset}

    alignment = int(kv.get("general.alignment", 32))
    data_start = (cursor + alignment - 1) // alignment * alignment
    for name, entry in tensors.items():
        numel = 1
        for d in entry["shape"]:
            numel *= int(d)
        blocks = numel // QK_BLOCK
        begin = data_start + int(entry["offset"])
        entry["payload"] = raw[begin : begin + blocks * BLOCK_BYTES]
        entry["blocks"] = blocks
        entry["codebook"] = kv[codebook_key(name)]
    return {"version": version, "kv": kv, "tensors": tensors}


def build_tensor_entry(values: np.ndarray, importance: np.ndarray, patterns, shape) -> dict:
    header, payload, blocks = encode_tensor(values, importance, patterns)
    return {"codebook": header, "payload": payload, "shape": shape, "blocks": blocks}


def decode_from_file(path: Path, name: str) -> np.ndarray:
    parsed = read_gguf(path)
    entry = parsed["tensors"][name]
    return decode_tensor(entry["codebook"], entry["payload"], entry["blocks"])

