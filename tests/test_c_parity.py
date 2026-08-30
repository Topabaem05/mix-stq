from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ltc_format import (
    BLOCK_BYTES,
    QK_BLOCK,
    decode_tensor,
    encode_tensor,
    fit_codebook,
    stq_codebook,
)
from remote_weights import fetch_tensor, read_header

URL = "https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00001-of-00003.safetensors"
NAME = "model.layers.0.mlp.experts.0.gate_proj.weight"
SAMPLE = QK_BLOCK * 128

header = read_header(URL)
raw = fetch_tensor(URL, header, NAME).astype(np.float64).reshape(-1)
values = raw[:SAMPLE]
rng = np.random.default_rng(4242)
importance = np.abs(rng.normal(0.0, 1.0, values.size)) ** 2 + 1e-6

failures = []
cases = {
    "learned": fit_codebook(values, importance),
    "stq_3to4": stq_codebook(),
}

for label, patterns in cases.items():
    codebook, payload, blocks = encode_tensor(values, importance, patterns)
    python_out = decode_tensor(codebook, payload, blocks)

    with tempfile.TemporaryDirectory() as tmp:
        cb_path = Path(tmp) / "cb.bin"
        pl_path = Path(tmp) / "pl.bin"
        cb_path.write_bytes(codebook)
        pl_path.write_bytes(payload)
        result = subprocess.run(
            ["./ltc_dequant", str(cb_path), str(pl_path), str(blocks * QK_BLOCK)],
            capture_output=True,
        )
    if result.returncode != 0:
        failures.append("%s: C decoder exited %d: %s" % (
            label, result.returncode, result.stderr.decode()[:200]))
        continue

    c_out = np.frombuffer(result.stdout, dtype="<f4")
    if c_out.size != python_out.size:
        failures.append("%s: size mismatch C=%d py=%d" % (label, c_out.size, python_out.size))
        continue

    python_f32 = python_out.astype(np.float32)
    exact = bool(np.array_equal(c_out, python_f32))
    max_delta = float(np.max(np.abs(c_out.astype(np.float64) - python_out)))
    nonzero_frac = float(np.mean(c_out != 0.0))
    print("%-10s blocks=%3d  bit_exact=%-5s  max_delta=%.3e  nonzero=%.3f" % (
        label, blocks, exact, max_delta, nonzero_frac))
    if not exact:
        failures.append("%s: C and Python differ (max delta %.3e)" % (label, max_delta))

print()
print("=== struct layout and bit packing checks ===")
probe = subprocess.run(["./ltc_dequant"], capture_output=True)
if probe.returncode != 2:
    failures.append("usage path should exit 2, got %d" % probe.returncode)
else:
    print("usage guard exits 2: PASS")

codebook, payload, blocks = encode_tensor(values, importance, cases["learned"])
if len(payload) != blocks * BLOCK_BYTES:
    failures.append("payload geometry wrong")
else:
    print("payload = %d blocks x %d bytes = %d: PASS" % (blocks, BLOCK_BYTES, len(payload)))

with tempfile.TemporaryDirectory() as tmp:
    cb_path = Path(tmp) / "cb.bin"
    pl_path = Path(tmp) / "pl.bin"
    cb_path.write_bytes(codebook)
    pl_path.write_bytes(payload[: BLOCK_BYTES * (blocks - 1)])
    truncated = subprocess.run(
        ["./ltc_dequant", str(cb_path), str(pl_path), str(blocks * QK_BLOCK)],
        capture_output=True,
    )
if truncated.returncode == 0:
    failures.append("truncated payload should fail, but exited 0")
else:
    print("truncated payload rejected (exit %d): PASS" % truncated.returncode)

print()
if failures:
    print("FAIL")
    for item in failures:
        print("  " + item)
    raise SystemExit(1)
print("PASS: C decoder is bit-exact with the Python reference")
print("  both learned and 3:4 codebooks verified on real OLMoE weights")
print("  %d weights per case" % (blocks * QK_BLOCK))
print("  fp16 scale conversion, 5-bit unaligned unpacking, base-3 codebook decode all match")

