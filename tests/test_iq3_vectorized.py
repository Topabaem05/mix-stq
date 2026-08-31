from __future__ import annotations

import time

import iq3_reference as reference
import iq3_vectorized as vectorized
import numpy as np
import torch

rng = np.random.default_rng(0)
cases = [
    (
        rng.standard_normal((4, 512)).astype(np.float32),
        (rng.random(512) + 0.1).astype(np.float32),
    ),
    (
        rng.standard_normal((2, 768)).astype(np.float32),
        (rng.random(768) + 0.1).astype(np.float32),
    ),
]
cases[0][0][0, :32] = 0
cases[0][0][0, 32:40] = np.asarray([-1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)

reference_seconds = 0.0
vectorized_seconds = 0.0
results = []
for matrix, importance in cases:
    started = time.perf_counter()
    expected, expected_error = reference.quantize_rows_reference(matrix, importance)
    reference_seconds += time.perf_counter() - started
    torch_matrix = torch.tensor(matrix.tolist(), dtype=torch.float32)
    torch_importance = torch.tensor(importance.tolist(), dtype=torch.float32)
    started = time.perf_counter()
    actual, actual_error = vectorized.quantize_rows_reference_torch(
        torch_matrix, torch_importance, chunk=4
    )
    vectorized_seconds += time.perf_counter() - started
    actual_array = np.asarray(actual.tolist(), dtype=np.float32)
    maximum_delta = float(np.max(np.abs(expected - actual_array)))
    if abs(expected_error - actual_error) > 1e-6:
        raise AssertionError(
            "relative errors differ: %.12f != %.12f"
            % (expected_error, actual_error)
        )
    if maximum_delta >= 1e-5:
        raise AssertionError("maximum reconstruction delta is %.12g" % maximum_delta)
    results.append((matrix.shape, expected_error, actual_error, maximum_delta))

for shape, expected_error, actual_error, maximum_delta in results:
    print("shape: %s" % (shape,))
    print("reference relative error: %.12f" % expected_error)
    print("vectorized relative error: %.12f" % actual_error)
    print("max elementwise delta: %.12g" % maximum_delta)
tail_input = torch.tensor([[float(index) for index in range(260)]], dtype=torch.float64)
tail_importance = torch.ones(256, dtype=torch.float64)
tail_output, _ = vectorized.quantize_rows_reference_torch(
    tail_input, tail_importance, chunk=1
)
if tail_output.shape != tail_input.shape or tail_output.dtype != tail_input.dtype:
    raise AssertionError("shape or dtype changed")
if not torch.equal(tail_output[:, 256:], tail_input[:, 256:]):
    raise AssertionError("unquantized tail changed")
print("reference seconds: %.6f" % reference_seconds)
print("vectorized seconds: %.6f" % vectorized_seconds)
print("speedup: %.3fx" % (reference_seconds / vectorized_seconds))
print("PASS: IQ3_XXS vectorized encoder matches the reference")
