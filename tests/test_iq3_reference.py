from __future__ import annotations

import json
from pathlib import Path

import iq3_reference as reference
import numpy as np
import torch
import torch_iq2 as approximation

rng = np.random.default_rng(0)
matrix = rng.standard_normal((8, 512)).astype(np.float32)
importance = (rng.random(512) + 0.1).astype(np.float32)

table_payload = json.loads(
    (Path(reference.__file__).with_name("tier_tables.json")).read_text(encoding="utf-8")
)
raw_grid = np.asarray(table_payload["iq3xxs_grid"], dtype=np.int16)
forward_map, _ = reference.build_grid_map(raw_grid)

reference_reconstruction, reference_error = reference.quantize_rows_reference(matrix, importance)
reference_indices = []
for row in range(matrix.shape[0]):
    for start in range(0, matrix.shape[1], reference.QK_K):
        state = reference._quantize_superblock_state(
            matrix[row, start : start + reference.QK_K],
            importance[start : start + reference.QK_K],
        )
        reference_indices.extend(state[1].tolist())

if reference_reconstruction.shape != matrix.shape:
    raise AssertionError("reference changed the input shape")
if any(index < 0 or index >= len(raw_grid) for index in reference_indices):
    raise AssertionError("reference emitted an invalid grid index")

reference_levels = (raw_grid // 4 - 1) // 2
reference_packed = (
    reference_levels[:, 0]
    | (reference_levels[:, 1] << 3)
    | (reference_levels[:, 2] << 6)
    | (reference_levels[:, 3] << 9)
)
if any(forward_map[int(reference_packed[index])] != index for index in reference_indices):
    raise AssertionError("reference emitted a lane absent from the IQ3_XXS grid")
if not np.isfinite(reference_error) or not 0 < reference_error < 1:
    raise AssertionError("reference relative error is outside (0, 1)")

torch_matrix = torch.tensor(matrix.tolist(), dtype=torch.float32)
torch_importance = torch.tensor(importance.tolist(), dtype=torch.float32)
approximation_reconstruction, approximation_error = approximation.quantize_rows(
    torch_matrix, torch_importance, tier="iq3_xxs"
)

body = torch_matrix.reshape(-1, approximation.QK_BLOCK)
weight_body = (
    torch_importance.reshape(1, -1)
    .expand(torch_matrix.shape[0], -1)
    .reshape(-1, approximation.QK_BLOCK)
)
lanes = body.reshape(-1, 4)
lane_weights = weight_body.reshape(-1, 4)
table = approximation.grid("cpu", "iq3_xxs")
approximation_indices, approximation_steps = approximation._solve_chunk(
    lanes.abs(), lane_weights, table, table.square()
)
approximation_levels = np.asarray(
    ((table[approximation_indices].to(torch.int16) // 4 - 1) // 2).tolist(), dtype=np.int16
)
approximation_packed = (
    approximation_levels[:, 0]
    | (approximation_levels[:, 1] << 3)
    | (approximation_levels[:, 2] << 6)
    | (approximation_levels[:, 3] << 9)
)
off_grid_lanes = int(np.count_nonzero(forward_map[approximation_packed] < 0))

negative = np.asarray(lanes.tolist(), dtype=np.float32) < 0
odd_parity_groups = np.count_nonzero(np.sum(negative.reshape(-1, 8), axis=1) % 2)
parity_invalid_lanes = int(2 * odd_parity_groups)
step_groups = np.asarray(approximation_steps.tolist(), dtype=np.float32).reshape(-1, 8)
nonshared_scale_groups = int(np.count_nonzero(np.ptp(step_groups, axis=1) > 1e-7))

ratio = approximation_error / reference_error
gap = approximation_error - reference_error

print("neighbour repair: brute-force weighted L2 over all 256 grid entries")
print("reference relative error: %.12f" % reference_error)
print("approximation relative error: %.12f" % approximation_error)
print("approximation/reference ratio: %.12f" % ratio)
print("approximation-reference gap: %.12f" % gap)
print("approximation off-grid magnitude lanes: %d / %d" % (off_grid_lanes, len(lanes)))
print("approximation parity-invalid lanes: %d / %d" % (parity_invalid_lanes, len(lanes)))
print(
    "approximation nonshared-scale 32-value groups: %d / %d"
    % (nonshared_scale_groups, len(step_groups))
)
print("PASS: IQ3_XXS reference audit completed")
