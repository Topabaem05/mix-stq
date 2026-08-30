# MIX-STQ — Learned Ternary Codebook quantization

MIX-STQ replaces the fixed 3:4 sparse ternary pattern set of `STQ1_0` with a
per-tensor **learned** codebook at the same bit width. It is a drop-in
replacement: the block layout is byte-identical to `STQ1_0`, so only the
32-byte codebook changes.

## Result

| arm | ppl | logit KL | top-1 | margin | router |
|---|---:|---:|---:|---:|---:|
| dense | 13.02 | 0 | 1.000 | 1.000 | 1.000 |
| mixed STQ1_0 | 31.67 | 0.959 | 0.561 | 0.871 | 0.704 |
| **mixed LTC** | **27.26** | **0.824** | **0.593** | **0.897** | **0.731** |

Same 1.875 bpw, same layer placement, 96 documents / 18,394 tokens.
Paired bootstrap on perplexity: **+4.410, 95% CI [+3.702, +5.221]**.

## Why the fixed pattern set costs quality

`STQ1_0` forces exactly one zero in every group of four, pinning the zero rate
at 0.250. Measured on real weights, the unconstrained ternary optimum wants
about 0.31:

| model | structure | natural zero rate |
|---|---|---:|
| OLMoE-1B-7B | MoE expert | 0.313 |
| Qwen1.5-MoE-A2.7B | MoE routed / shared | 0.312 / 0.315 |
| Qwen3.8-27B | dense MLP | 0.312 |

Five bits address 32 entries; the ternary 4-tuple space holds 81. Choosing
which 32 per tensor is what LTC does.

## Layout

```
block: 42 bytes                 identical to STQ1_0
  fp16 scale                    2 bytes
  64 groups x 5 bits            40 bytes

per-tensor GGUF metadata
  ltc.codebook.<tensor_name>    uint8[32], base-3 pattern indices
```

Restricting the codebook to the 3:4 subset reproduces `STQ1_0` bit-for-bit,
which the test suite pins.

## Status

| component | state |
|---|---|
| encoder, codebook fitting | done |
| GGUF container round trip | done, lossless |
| `dequantize_row_ltc1_0` (C) | done, bit-exact vs Python |
| `ggml_vec_dot_ltc1_0` (C) | done, 6e-07 relative, 2.37x over dequantize path |
| ggml type registration, loader wiring | not started |
| SIMD (AVX2/NEON), CUDA/Metal | not started |

## What was disproved

Two ideas were tested and rejected rather than quietly dropped.

- **LTC does not replace IQ2_XXS.** At 1.9688 bpw IQ2_XXS is 2.2% better on
  perplexity with the CI excluding zero. LTC's place is at the STQ bit width.
- **Sensitivity-driven layer selection does not help.** A deliberately worst
  layer assignment was statistically indistinguishable from the best one
  (all CIs contained zero). MoE expert tensors are too homogeneous for the
  signal to matter.

## Layout of this repository

```
src/mixstq/   codecs, codebook fitting, GGUF container, allocator, harness
csrc/         reference C decoder and fused vec_dot
tests/        invariants, container round trip, C parity
docs/         per-stage research records with measured numbers
artifacts/    imatrix and evaluation outputs
```

## Verification

```bash
python3 -m pytest tests -q          # invariants, round trip, C parity
clang -O2 -Wall -Wextra -o ltc_dequant csrc/ltc_dequant.c
clang -O2 -Wall -Wextra -o ltc_vecdot csrc/ltc_vecdot.c
ruff check .
```

GPU stages use `src/mixstq/vast_control.py`, which defaults to a dry run and
refuses to spend without `--confirm`.

## Not measured

Task accuracy (MMLU and similar), llama.cpp end-to-end inference, GPU kernel
throughput, and QAT interaction are all open. Perplexity, logit KL, top-1,
margin, and router agreement are what the numbers above cover.

## License

MIT.
