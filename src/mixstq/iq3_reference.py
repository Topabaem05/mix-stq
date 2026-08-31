from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

QK_K = 256
K_MAX_Q = 8
GROUP_MAX_EPS_IQ3_XXS = np.float32(1e-8)


def _odd_grid(grid):
    values = np.asarray(grid, dtype=np.int16)
    if np.max(values) > 15:
        values = values // 4
    return values


def _packed_levels(levels):
    return (
        levels[..., 0]
        | (levels[..., 1] << 3)
        | (levels[..., 2] << 6)
        | (levels[..., 3] << 9)
    )


def build_grid_map(grid):
    odd_grid = _odd_grid(grid)
    levels = (odd_grid - 1) // 2
    packed = _packed_levels(levels)
    forward_map = np.full(4096, -1, dtype=np.int16)
    forward_map[packed] = np.arange(len(odd_grid), dtype=np.int16)
    all_grid_indices = np.arange(len(odd_grid), dtype=np.int16)
    neighbour_lookup = {
        u: all_grid_indices for u in range(len(forward_map)) if forward_map[u] < 0
    }
    return forward_map, neighbour_lookup


@lru_cache(maxsize=1)
def _tables():
    payload = json.loads(
        Path(__file__).with_name("tier_tables.json").read_text(encoding="utf-8")
    )
    raw_grid = np.asarray(payload["iq3xxs_grid"], dtype=np.float32)
    odd_grid = _odd_grid(raw_grid).astype(np.float32)
    forward_map, neighbour_lookup = build_grid_map(raw_grid)
    return raw_grid, odd_grid, forward_map, neighbour_lookup


def _nearest_int(values):
    return np.rint(values).astype(np.int16)


def _best_grid_index(u, x_lane, neighbour_weight, scale):
    _, odd_grid, _, neighbour_lookup = _tables()
    candidates = neighbour_lookup[int(u)]
    differences = np.float32(scale) * odd_grid[candidates] - x_lane
    distances = np.sum(neighbour_weight * differences * differences, axis=1, dtype=np.float32)
    return int(candidates[int(np.argmin(distances))])


def _repair_lane(levels, x_lane, neighbour_weight, scale):
    _, odd_grid, forward_map, _ = _tables()
    u = int(_packed_levels(levels))
    grid_index = int(forward_map[u])
    if grid_index < 0:
        grid_index = _best_grid_index(u, x_lane, neighbour_weight, scale)
    repaired = ((odd_grid[grid_index] - 1) / 2).astype(np.int16)
    return repaired, grid_index, int(forward_map[u]) >= 0


def _quantize_subblock(xb, quant_weights, sigma2):
    _, _, forward_map, _ = _tables()
    if quant_weights is None:
        weight = np.float32(xb * xb)
    else:
        weight = np.float32(quant_weights * np.sqrt(np.float32(sigma2 + xb * xb)))
    neighbour_weight = np.sqrt(weight).astype(np.float32)
    xval = np.empty(32, dtype=np.float32)
    block_signs = np.zeros(4, dtype=np.uint8)
    for group in range(4):
        start = 8 * group
        values = xb[start : start + 8]
        negative = values < 0
        xval[start : start + 8] = np.abs(values)
        sign_bits = sum(int(flag) << i for i, flag in enumerate(negative))
        if int(np.count_nonzero(negative)) % 2:
            costs = weight[start : start + 8] * values * values
            minimum = int(np.argmin(costs))
            xval[start + minimum] = -xval[start + minimum]
            sign_bits ^= 1 << minimum
        block_signs[group] = sign_bits & 127

    maximum = np.max(xval)
    levels = np.zeros(32, dtype=np.int16)
    if maximum < GROUP_MAX_EPS_IQ3_XXS:
        return np.float32(0), levels, block_signs

    best = np.float32(0)
    scale = np.float32(maximum / np.float32(2 * K_MAX_Q - 1))
    selected_on_grid = np.ones(8, dtype=bool)
    for search_index in range(-15, 16):
        inverse_scale = np.float32(
            np.float32(2 * K_MAX_Q - 1 + search_index * 0.2) / maximum
        )
        candidate_scale = np.float32(1 / inverse_scale)
        candidate_levels = np.empty(32, dtype=np.int16)
        candidate_on_grid = np.ones(8, dtype=bool)
        for lane in range(8):
            start = 4 * lane
            lane_levels = _nearest_int(
                np.float32(0.5) * (inverse_scale * xval[start : start + 4] - 1)
            )
            lane_levels = np.clip(lane_levels, 0, K_MAX_Q - 1).astype(np.int16)
            repaired, _, on_grid = _repair_lane(
                lane_levels,
                xval[start : start + 4],
                neighbour_weight[start : start + 4],
                candidate_scale,
            )
            candidate_levels[start : start + 4] = repaired
            candidate_on_grid[lane] = on_grid
        quant_levels = np.float32(2 * candidate_levels + 1)
        sumqx = np.sum(weight * xval * quant_levels, dtype=np.float32)
        sumq2 = np.sum(weight * quant_levels * quant_levels, dtype=np.float32)
        if sumq2 > 0 and sumqx * sumqx > best * sumq2:
            scale = np.float32(sumqx / sumq2)
            best = np.float32(scale * sumqx)
            levels[:] = candidate_levels
            selected_on_grid[:] = candidate_on_grid

    if np.any(~selected_on_grid) and scale > 0:
        inverse_scale = np.float32(1 / scale)
        for lane in np.flatnonzero(~selected_on_grid):
            start = 4 * int(lane)
            lane_levels = _nearest_int(
                np.float32(0.5) * (inverse_scale * xval[start : start + 4] - 1)
            )
            lane_levels = np.clip(lane_levels, 0, K_MAX_Q - 1).astype(np.int16)
            repaired, _, _ = _repair_lane(
                lane_levels,
                xval[start : start + 4],
                neighbour_weight[start : start + 4],
                scale,
            )
            levels[start : start + 4] = repaired
        quant_levels = np.float32(2 * levels + 1)
        sumqx = np.sum(weight * xval * quant_levels, dtype=np.float32)
        sumq2 = np.sum(weight * quant_levels * quant_levels, dtype=np.float32)
        if sumq2 > 0:
            scale = np.float32(sumqx / sumq2)

    if scale < 0:
        scale = np.float32(-scale)
        block_signs[:] = np.bitwise_not(block_signs) & 127

    for lane in range(8):
        start = 4 * lane
        u = int(_packed_levels(levels[start : start + 4]))
        if forward_map[u] < 0:
            raise RuntimeError("IQ3_XXS repair emitted an off-grid lane")
    return scale, levels, block_signs


def _decode_signs(sign_code):
    decoded = int(sign_code)
    if decoded.bit_count() % 2:
        decoded |= 128
    return np.asarray([-1 if decoded & (1 << i) else 1 for i in range(8)], dtype=np.float32)


def _quantize_superblock_state(x, quant_weights):
    raw_grid, _, forward_map, _ = _tables()
    block = np.asarray(x, dtype=np.float32).reshape(-1)
    if block.size != QK_K:
        raise ValueError("IQ3_XXS superblocks must contain 256 values")
    weights = None
    if quant_weights is not None:
        weights = np.asarray(quant_weights, dtype=np.float32).reshape(-1)
        if weights.size != QK_K:
            raise ValueError("quant_weights must contain 256 values")

    sumx2 = np.sum(block * block, dtype=np.float32)
    sigma2 = np.float32(np.float32(2) * sumx2 / QK_K)
    scales = np.zeros(8, dtype=np.float32)
    lane_indices = np.zeros(64, dtype=np.int16)
    sign_codes = np.zeros(32, dtype=np.uint8)
    levels_by_subblock = np.zeros((8, 32), dtype=np.int16)
    for subblock in range(8):
        start = 32 * subblock
        block_weights = None if weights is None else weights[start : start + 32]
        scale, levels, block_signs = _quantize_subblock(
            block[start : start + 32], block_weights, sigma2
        )
        scales[subblock] = scale
        levels_by_subblock[subblock] = levels
        sign_codes[4 * subblock : 4 * subblock + 4] = block_signs
        for lane in range(8):
            lane_start = 4 * lane
            u = int(_packed_levels(levels[lane_start : lane_start + 4]))
            lane_indices[8 * subblock + lane] = forward_map[u]

    reconstruction = np.zeros(QK_K, dtype=np.float32)
    scale_levels = np.zeros(8, dtype=np.int16)
    max_scale = np.max(scales)
    packed_scales_and_signs = np.zeros(8, dtype=np.uint32)
    for subblock in range(8):
        codes = sign_codes[4 * subblock : 4 * subblock + 4]
        packed_scales_and_signs[subblock] = (
            int(codes[0])
            | (int(codes[1]) << 7)
            | (int(codes[2]) << 14)
            | (int(codes[3]) << 21)
        )
    if max_scale > 0:
        base_scale = np.float32(max_scale / 31)
        stored_base_scale = np.float32(np.float16(np.float32(base_scale * np.float32(1.0125))))
        inverse_base_scale = np.float32(1 / base_scale)
        scale_levels[:] = np.clip(
            _nearest_int(np.float32(0.5) * (inverse_base_scale * scales - 1)), 0, 15
        )
        for subblock in range(8):
            packed_scales_and_signs[subblock] |= np.uint32(int(scale_levels[subblock]) << 28)
            block_scale = np.float32(
                stored_base_scale
                * np.float32(np.float32(0.5) + scale_levels[subblock])
                * np.float32(0.5)
            )
            for group in range(4):
                signs = _decode_signs(sign_codes[4 * subblock + group])
                for half in range(2):
                    lane = 8 * subblock + 2 * group + half
                    output_start = 32 * subblock + 8 * group + 4 * half
                    reconstruction[output_start : output_start + 4] = (
                        block_scale * raw_grid[lane_indices[lane]] * signs[4 * half : 4 * half + 4]
                    )
    return reconstruction, lane_indices, packed_scales_and_signs, scale_levels


def quantize_superblock_reference(x, quant_weights):
    block = np.asarray(x, dtype=np.float32).reshape(-1)
    weights = np.ones(QK_K, dtype=np.float32)
    if quant_weights is not None:
        weights = np.asarray(quant_weights, dtype=np.float32).reshape(-1)
    reconstruction, _, _, _ = _quantize_superblock_state(block, quant_weights)
    error = np.sum(weights * (block - reconstruction) ** 2, dtype=np.float64)
    energy = np.sum(weights * block**2, dtype=np.float64)
    return reconstruction, float(error / max(energy, 1e-30))


def quantize_rows_reference(matrix, importance):
    values = np.asarray(matrix)
    work = values.astype(np.float32, copy=False)
    if work.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    channel_weights = np.asarray(importance, dtype=np.float32).reshape(-1)
    rows, width = work.shape
    usable = width // QK_K * QK_K
    if channel_weights.size < usable:
        raise ValueError("importance is shorter than the quantized width")
    output = work.copy()
    for row in range(rows):
        for start in range(0, usable, QK_K):
            output[row, start : start + QK_K], _ = quantize_superblock_reference(
                work[row, start : start + QK_K], channel_weights[start : start + QK_K]
            )
    expanded_weights = np.broadcast_to(channel_weights[:usable], (rows, usable))
    error = np.sum(
        expanded_weights * (work[:, :usable] - output[:, :usable]) ** 2, dtype=np.float64
    )
    energy = np.sum(expanded_weights * work[:, :usable] ** 2, dtype=np.float64)
    relative_error = 0.0 if usable == 0 else float(error / max(energy, 1e-30))
    return output.astype(values.dtype, copy=False), relative_error
