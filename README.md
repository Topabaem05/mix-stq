# MIX-STQ — where low-bit quantization stops costing accuracy

A measurement study, not a codec proposal. It started as a learned ternary
codebook (LTC) meant to beat the fixed 3:4 pattern set of `STQ1_0`, and ended
somewhere else: the bit budget matters far more than the codebook, and one
standard ggml tier already holds full accuracy.

Artifacts and importance matrices: [topabaem/mix-stq-artifacts](https://huggingface.co/datasets/topabaem/mix-stq-artifacts)

## Headline result

Qwen3.8-27B, MMLU 140 + ARC-Challenge 60, paired McNemar plus bootstrap CI:

| arm | bpw | accuracy | vs bf16 | p | verdict |
|---|---:|---:|---:|---:|---|
| bf16 | 16 | 0.8050 | baseline | — | — |
| IQ2_XXS | 2.0625 | 0.7300 | −0.0750 | **0.0007** | loses |
| **IQ3_XXS** | **3.0625** | **0.7900** | −0.0150 | 0.5078 | **indistinguishable** |
| IQ3_S | 3.4375 | 0.7900 | −0.0150 | 0.4531 | indistinguishable |

**IQ3_XXS at 3.0625 bpw is the answer.** 5.2x compression, no measurable
accuracy cost, and it matches bf16 exactly on ARC-Challenge (0.967). Spending
another 0.375 bpw on IQ3_S buys nothing.

Confirmed on two architectures: OLMoE-1B-7B (MoE) and Qwen3.8-27B (dense).

## What did not survive measurement

Recorded rather than quietly dropped. Each has a dated research note in `docs/`.

| claim | fate |
|---|---|
| LTC beats the fixed pattern set | true on perplexity (−13.9%), **false on task accuracy** (delta 0.0000, p=1.0000) |
| LTC replaces IQ2_XXS | false: IQ2_XXS wins by 2.2% perplexity at equal bits |
| Sensitivity-driven layer allocation helps | false: worst assignment indistinguishable from best |
| Mixing tiers by layer helps | false: what sets accuracy is the *lowest* tier present, not the average bpw |
| 2.06–2.69 bpw is a plateau | artifact of OLMoE's low baseline; on 27B IQ2_XXS loses significantly |
| IQ3_S beats bf16 | did not reproduce on 27B |

The LTC line is closed. A learned 32-entry codebook loses to a fixed 256-entry
grid by 12.5 points at the same 4-value lane structure, and growing the codebook
to 256 entries would cost eight address bits and collapse into IQ3_XXS.

## Why reconstruction error misleads

Weighted mean error is a poor proxy for task accuracy in two specific ways.

Below a threshold it saturates: from 2.06 to 2.56 bpw error improves 36% while
accuracy drifts *down* a point. Only the 80% drop at 3.06 buys anything.

Under mixed allocation it inverts: an arm with 28% lower error than IQ2_S scored
lower, because averaging hides the bottleneck tensors.

## Encoder

The scale is solved in closed form rather than searched. For a fixed codebook
entry the per-lane objective is quadratic in the scale:

```
objective(step) = quadratic * step^2 - 2 * linear * step
step*           = linear / quadratic
minimum         = -linear^2 / quadratic
```

Taking argmin of the minimum over entries replaces a 144-point grid search. It
is 84–102x faster *and* lands lower error on every tier, since the grid was
missing the optimum. This turned a 195-minute sweep into 13 minutes.

Supported tiers, all extracted from `ggml-common.h` rather than invented:

| tier | bpw | grid | lane |
|---|---:|---:|---|
| IQ2_XXS | 2.0625 | 256 | 8 values |
| IQ2_XS | 2.3125 | 512 | 8 values |
| IQ2_S | 2.5625 | 1024 | 8 values |
| IQ3_XXS | 3.0625 | 256 | 4 values |
| IQ3_S | 3.4375 | 512 | 4 values |

## Harness

Two scoring protocols, because they answer different questions.

`eval_tasks.py` picks the answer by comparing per-letter log-probabilities.
Cheap and low-variance, but not how published numbers are produced.

`eval_generative.py` generates an answer and parses it, matching the protocol
behind the official GPQA Diamond figure. Batched, since 198 items at 2048 new
tokens each is otherwise hours per arm.

Both walk MoE fused expert parameters and dense per-projection weights, and both
raise if a plan quantizes nothing — a guard added after a silent no-op produced
a plausible-looking null result.

## What is not verified

- **No GGUF checkpoint exists.** Accuracy is measured by quantizing weights in
  PyTorch, not by writing a file and running llama.cpp. GGUF round-trip and C
  decoder parity are done for LTC only, not for the IQ tiers.
- Attention projections and embeddings are untouched; only MLP tensors are
  quantized, so real deployment size would differ.
- 200-item samples give a paired CI half-width around ±3–6 points, so only the
  differences marked significant above are resolved.
- Generation quality is unmeasured; everything here is multiple choice.

## Repository layout

```
src/mixstq/   codecs, codebook fitting, GGUF container, allocator, harnesses
csrc/         reference C decoder and fused vec_dot
tests/        invariants, container round trip, C parity, encoder equivalence
docs/         dated research records, each with a verification log
artifacts/    importance matrices and evaluation outputs
```

## Verification

```bash
python3 scripts/run_tests.py --offline   # tests are standalone scripts, not pytest
python3 -m ruff check .
clang -O2 -Wall -Wextra -o ltc_dequant csrc/ltc_dequant.c
clang -O2 -Wall -Wextra -o ltc_vecdot csrc/ltc_vecdot.c
```

GPU stages use `src/mixstq/vast_control.py`, which defaults to a dry run and
refuses to spend without `--confirm`.

