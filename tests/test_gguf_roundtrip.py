from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from gguf_ltc import (
    GGML_TYPE_LTC1_0,
    build_tensor_entry,
    codebook_key,
    decode_from_file,
    read_gguf,
    write_gguf,
)
from ltc_format import (
    BLOCK_BYTES,
    CODEBOOK_BYTES,
    QK_BLOCK,
    bits_per_weight,
    decode_tensor,
    encode_tensor,
    fit_codebook,
    stored_bytes,
    stq_codebook,
)
from remote_weights import fetch_tensor, read_header

URL = "https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00001-of-00003.safetensors"
NAME = "model.layers.0.mlp.experts.0.gate_proj.weight"
SAMPLE = QK_BLOCK * 64

failures = []
header = read_header(URL)
raw = fetch_tensor(URL, header, NAME).astype(np.float64).reshape(-1)
values = raw[:SAMPLE]
rng = np.random.default_rng(909)
importance = np.abs(rng.normal(0.0, 1.0, values.size)) ** 2 + 1e-6

patterns = fit_codebook(values, importance)
cb, payload, blocks = encode_tensor(values, importance, patterns)

if len(cb) != CODEBOOK_BYTES:
    failures.append("codebook header is %d bytes, expected %d" % (len(cb), CODEBOOK_BYTES))
if len(payload) != blocks * BLOCK_BYTES:
    failures.append("payload size mismatch")

direct = decode_tensor(cb, payload, blocks)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "ltc.gguf"
    entry = build_tensor_entry(values, importance, patterns, [SAMPLE])
    stats = write_gguf(path, {NAME: entry})
    parsed = read_gguf(path)

    if parsed["version"] != 3:
        failures.append("gguf version %s" % parsed["version"])
    if parsed["kv"]["ltc.block_size"] != QK_BLOCK:
        failures.append("block size metadata wrong")
    if parsed["kv"]["ltc.block_bytes"] != BLOCK_BYTES:
        failures.append("block bytes metadata wrong")
    if parsed["tensors"][NAME]["type"] != GGML_TYPE_LTC1_0:
        failures.append("ggml type not preserved")
    if parsed["kv"][codebook_key(NAME)] != cb:
        failures.append("codebook did not survive the container")
    if parsed["tensors"][NAME]["payload"] != payload:
        failures.append("payload did not survive the container")

    from_file = decode_from_file(path, NAME)
    if not np.array_equal(from_file, direct):
        failures.append("container decode differs from direct decode")

    file_bytes = stats["bytes"]

expected = stored_bytes(SAMPLE)
bpw = bits_per_weight(SAMPLE)
energy = float(np.sum(importance * values ** 2))
error = float(np.sum(importance * (values - direct) ** 2)) / energy

stq_cb, stq_payload, stq_blocks = encode_tensor(values, importance, stq_codebook())
stq_direct = decode_tensor(stq_cb, stq_payload, stq_blocks)
stq_error = float(np.sum(importance * (values - stq_direct) ** 2)) / energy

print("=== GGUF container round trip on real OLMoE weights ===")
print("tensor            %s" % NAME)
print("weights           %d (%d blocks)" % (SAMPLE, blocks))
print("codebook header   %d bytes" % len(cb))
print("payload           %d bytes" % len(payload))
print("logical stored    %d bytes -> %.4f bpw" % (expected, bpw))
print("gguf file total   %d bytes (includes header + alignment padding)" % file_bytes)
print()
print("LTC  relative error %.4f" % error)
print("STQ  relative error %.4f" % stq_error)
print("LTC vs STQ          %+.1f%%" % (-100 * (1 - error / stq_error)))
print()

if failures:
    print("FAIL")
    for item in failures:
        print("  " + item)
    raise SystemExit(1)

print("PASS: all container invariants hold")
print("  codebook survives as uint8 KV metadata (32 bytes)")
print("  payload byte-identical through write/read")
print("  container decode == direct decode (exact)")
print("  ggml type id %d preserved" % GGML_TYPE_LTC1_0)
print("  block geometry metadata intact")

