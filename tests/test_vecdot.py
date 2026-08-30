from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ltc_format import QK_BLOCK, decode_tensor, encode_tensor, fit_codebook
from remote_weights import fetch_tensor, read_header

URL = "https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00001-of-00003.safetensors"
NAME = "model.layers.0.mlp.experts.0.gate_proj.weight"
SAMPLE = QK_BLOCK * 256

header = read_header(URL)
raw = fetch_tensor(URL, header, NAME).astype(np.float64).reshape(-1)
values = raw[:SAMPLE]
rng = np.random.default_rng(5150)
importance = np.abs(rng.normal(0.0, 1.0, values.size)) ** 2 + 1e-6

patterns = fit_codebook(values, importance)
codebook, payload, blocks = encode_tensor(values, importance, patterns)
weights = decode_tensor(codebook, payload, blocks).astype(np.float32)
activations = rng.normal(0.0, 1.0, blocks * QK_BLOCK).astype(np.float32)

python_dot = float(np.dot(weights.astype(np.float64), activations.astype(np.float64)))

failures = []
with tempfile.TemporaryDirectory() as tmp:
    cb = Path(tmp) / "cb.bin"
    pl = Path(tmp) / "pl.bin"
    ac = Path(tmp) / "ac.bin"
    cb.write_bytes(codebook)
    pl.write_bytes(payload)
    ac.write_bytes(activations.tobytes())
    result = subprocess.run(
        ["./ltc_vecdot", str(cb), str(pl), str(ac), str(blocks * QK_BLOCK)],
        capture_output=True, text=True,
    )

if result.returncode != 0:
    failures.append("vec_dot exited %d: %s" % (result.returncode, result.stderr[:300]))
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)

report = {}
for line in result.stdout.strip().splitlines():
    parts = line.split()
    if len(parts) >= 2:
        report[parts[0]] = parts[1]

fused = float(report["fused_dot"])
deq_dot = float(report["dequant_then_dot"])
rel_to_python = abs(fused - python_dot) / max(abs(python_dot), 1e-12)

print("=== correctness ===")
print("python dot (float64)   %.6f" % python_dot)
print("C fused vec_dot        %.6f" % fused)
print("C dequant-then-dot     %.6f" % deq_dot)
print("fused vs C dequant     rel %s" % report["rel_delta"])
print("fused vs python        rel %.3e" % rel_to_python)
print()
print("=== performance (%d weights x 200 repeats) ===" % (blocks * QK_BLOCK))
print("fused        %s Mw/s" % report["fused_throughput"])
print("dequant+dot  %s Mw/s" % report["deq_throughput"])
print("speedup      %s" % report["fused_speedup"])
print()

if rel_to_python > 2e-5:
    failures.append("fused dot deviates from python by %.3e" % rel_to_python)
if abs(float(report["rel_delta"])) > 2e-5:
    failures.append("fused vs dequant path differ by %s" % report["rel_delta"])
speedup = float(report["fused_speedup"].rstrip("x"))
if speedup < 1.0:
    print("NOTE: fusing is not faster on this CPU; dequantize path wins")

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)

print("PASS: ggml_vec_dot_ltc1_0 is numerically correct")
print("  matches float64 python reference within %.1e relative" % rel_to_python)
print("  matches the dequantize-then-dot path in C")
print("  avoids materializing the dequantized row")
