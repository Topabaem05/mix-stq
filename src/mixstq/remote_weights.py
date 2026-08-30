from __future__ import annotations

import json
import struct
import subprocess

import numpy as np

DTYPE_MAP = {
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (None, 2),
}


def _curl_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    result = subprocess.run(
        ["curl", "-sS", "-L", "-m", str(timeout), "-H", "Range: bytes=%d-%d" % (start, end), url],
        capture_output=True,
        check=True,
    )
    return result.stdout


def read_header(url: str) -> dict:
    prefix = _curl_range(url, 0, 7)
    if len(prefix) < 8:
        raise RuntimeError("could not read safetensors length prefix")
    header_len = struct.unpack("<Q", prefix[:8])[0]
    raw = _curl_range(url, 8, 8 + header_len - 1)
    return json.loads(raw.decode("utf-8"))


def bf16_to_float32(raw: bytes) -> np.ndarray:
    as_u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (as_u16 << 16).view(np.float32)


def fetch_tensor(url: str, header: dict, name: str) -> np.ndarray:
    entry = header[name]
    dtype_name = entry["dtype"]
    if dtype_name not in DTYPE_MAP:
        raise RuntimeError("unsupported dtype " + dtype_name)
    data_start = 8 + struct.unpack("<Q", _curl_range(url, 0, 7)[:8])[0]
    begin, end = entry["data_offsets"]
    raw = _curl_range(url, data_start + begin, data_start + end - 1)
    expected = end - begin
    if len(raw) != expected:
        raise RuntimeError("short read: got %d want %d" % (len(raw), expected))
    if dtype_name == "BF16":
        values = bf16_to_float32(raw)
    else:
        np_dtype, _ = DTYPE_MAP[dtype_name]
        values = np.frombuffer(raw, dtype=np_dtype).astype(np.float32)
    return values.reshape(entry["shape"])

