from __future__ import annotations

import numpy as np
from codebook import ALL_PATTERNS, STQ_PATTERNS, assign, encode_fixed
from stq_codec import stq_bits_per_weight

failures = []

if abs(stq_bits_per_weight() - 1.3125) > 1e-12:
    failures.append("bpw mismatch: %r" % stq_bits_per_weight())

if ALL_PATTERNS.shape[0] != 81:
    failures.append("ternary 4-tuple space must be 81, got %d" % ALL_PATTERNS.shape[0])

if STQ_PATTERNS.shape[0] != 32:
    failures.append("3:4 pattern count must be 32, got %d" % STQ_PATTERNS.shape[0])

zeros_per_pattern = np.sum(STQ_PATTERNS == 0.0, axis=1)
if not np.all(zeros_per_pattern == 1):
    failures.append("every 3:4 pattern must contain exactly one zero")

if 32 > 2 ** 5:
    failures.append("codebook must fit in 5 bits")

rng = np.random.default_rng(3)
for trial in range(24):
    values = rng.laplace(0.0, 1.0, 1024)
    importance = np.abs(rng.normal(0, 1, 1024)) ** 2 + 1e-6
    scale, choice, err = encode_fixed(values, importance, STQ_PATTERNS)
    selected = STQ_PATTERNS[choice]
    if not np.all(np.sum(selected == 0.0, axis=1) == 1):
        failures.append("trial %d violated 3:4 invariant" % trial)
        break
    if scale <= 0.0:
        failures.append("trial %d produced non-positive scale %r" % (trial, scale))
        break
    if not np.isfinite(err):
        failures.append("trial %d produced non-finite error" % trial)
        break

rng = np.random.default_rng(5)
values = rng.laplace(0.0, 1.0, 4096)
importance = np.abs(rng.normal(0, 1, 4096)) ** 2 + 1e-6
scale_a, choice_a, err_a = encode_fixed(values, importance, STQ_PATTERNS)
scale_b, choice_b, err_b = encode_fixed(values, importance, STQ_PATTERNS)
if not (np.array_equal(choice_a, choice_b) and scale_a == scale_b):
    failures.append("encoder is not deterministic")

subset = ALL_PATTERNS[np.sum(ALL_PATTERNS == 0.0, axis=1) == 1]
_, _, err_subset = encode_fixed(values, importance, subset)
if abs(err_subset - err_a) > 1e-9:
    failures.append("LTC restricted to 3:4 must equal STQ1_0: %r vs %r" % (err_subset, err_a))

groups = values.reshape(-1, 4)
weights = importance.reshape(-1, 4)
choice_full, err_full = assign(groups, weights, ALL_PATTERNS, scale_a)
if float(np.sum(err_full)) > err_a + 1e-9:
    failures.append("81-pattern superset must never be worse than 32-pattern subset")

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("PASS: all %d structural invariants hold" % 8)
print("  stq bpw = 1.3125")
print("  ternary 4-tuple space = 81, 3:4 subset = 32 (fits 5 bits)")
print("  3:4 invariant preserved by encoder")
print("  encoder deterministic")
print("  LTC restricted to 3:4 == STQ1_0 (byte-compatible superset)")
print("  81-pattern superset never worse than 3:4 subset")
