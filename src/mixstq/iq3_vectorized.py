from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

QK_K = 256
K_MAX_Q = 8
GROUP_MAX_EPS_IQ3_XXS = 1e-8


@lru_cache(maxsize=None)
def _tables(device_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = json.loads(
        Path(__file__).with_name("tier_tables.json").read_text(encoding="utf-8")
    )
    device = torch.device(device_name)
    raw_grid = torch.tensor(payload["iq3xxs_grid"], dtype=torch.float32, device=device)
    odd_grid = torch.div(
        raw_grid.to(torch.int64), 4, rounding_mode="floor"
    ).to(torch.float32)
    levels = ((odd_grid.to(torch.int64) - 1) // 2).to(torch.int64)
    packed = (
        levels[:, 0]
        | (levels[:, 1] << 3)
        | (levels[:, 2] << 6)
        | (levels[:, 3] << 9)
    )
    forward_map = torch.full((4096,), -1, dtype=torch.int64, device=device)
    forward_map[packed] = torch.arange(len(odd_grid), device=device)
    return raw_grid, odd_grid, forward_map


def _nearest_int(values: torch.Tensor) -> torch.Tensor:
    lower = torch.floor(values)
    fraction = values - lower
    rounded = torch.where(
        fraction < 0.5,
        lower,
        torch.where(
            fraction > 0.5,
            lower + 1,
            torch.where((lower.to(torch.int64) & 1) == 0, lower, lower + 1),
        ),
    )
    return rounded.to(torch.int64)


def _packed_levels(levels: torch.Tensor) -> torch.Tensor:
    return (
        levels[..., 0]
        | (levels[..., 1] << 3)
        | (levels[..., 2] << 6)
        | (levels[..., 3] << 9)
    )


def _repair_lanes(
    levels: torch.Tensor,
    x_lanes: torch.Tensor,
    neighbour_weight: torch.Tensor,
    scales: torch.Tensor,
    odd_grid: torch.Tensor,
    forward_map: torch.Tensor,
    repair_batch: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = _packed_levels(levels)
    indices = forward_map[packed]
    on_grid = indices >= 0
    missing = torch.nonzero(~on_grid, as_tuple=False).flatten()
    for start in range(0, missing.numel(), repair_batch):
        positions = missing[start : start + repair_batch]
        differences = (
            scales[positions, None, None] * odd_grid[None, :, :]
            - x_lanes[positions, None, :]
        )
        distances = torch.sum(
            neighbour_weight[positions, None, :] * differences * differences,
            dim=2,
        )
        indices[positions] = torch.argmin(distances, dim=1)
    repaired = ((odd_grid[indices].to(torch.int64) - 1) // 2).to(torch.int64)
    return repaired, on_grid


def _force_even_parity(
    values: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = values.reshape(-1, 4, 8)
    grouped_weight = weight.reshape(-1, 4, 8)
    negative = grouped < 0
    xval = grouped.abs()
    powers = 1 << torch.arange(8, dtype=torch.int64, device=values.device)
    sign_codes = torch.sum(negative.to(torch.int64) * powers, dim=2)
    odd = torch.sum(negative, dim=2) % 2 == 1
    costs = grouped_weight * grouped * grouped
    minimum = torch.argmin(costs, dim=2)
    selected = torch.zeros_like(xval, dtype=torch.bool)
    selected.scatter_(2, minimum.unsqueeze(2), odd.unsqueeze(2))
    xval = torch.where(selected, -xval, xval)
    sign_codes = sign_codes ^ torch.where(odd, 1 << minimum, 0)
    return xval.reshape(-1, 32), sign_codes & 127


def _search_subblocks(
    values: torch.Tensor,
    quant_weights: torch.Tensor,
    sigma2: torch.Tensor,
    odd_grid: torch.Tensor,
    forward_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = quant_weights * torch.sqrt(sigma2[:, None] + values * values)
    neighbour_weight = torch.sqrt(weight)
    xval, sign_codes = _force_even_parity(values, weight)
    maximum = torch.amax(xval, dim=1)
    active = maximum >= GROUP_MAX_EPS_IQ3_XXS
    safe_maximum = torch.where(active, maximum, torch.ones_like(maximum))
    levels = torch.zeros_like(values, dtype=torch.int64)
    scale = safe_maximum / (2 * K_MAX_Q - 1)
    scale = torch.where(active, scale, torch.zeros_like(scale))
    best = torch.zeros_like(scale)
    selected_on_grid = torch.ones(
        (len(values), 8), dtype=torch.bool, device=values.device
    )
    x_lanes = xval.reshape(-1, 4)
    lane_weight = neighbour_weight.reshape(-1, 4)
    for search_index in range(-15, 16):
        inverse_scale = (2 * K_MAX_Q - 1 + search_index * 0.2) / safe_maximum
        candidate_scale = 1 / inverse_scale
        candidate_levels = _nearest_int(
            0.5 * (inverse_scale[:, None] * xval - 1)
        ).clamp_(0, K_MAX_Q - 1)
        repaired, on_grid = _repair_lanes(
            candidate_levels.reshape(-1, 4),
            x_lanes,
            lane_weight,
            candidate_scale.repeat_interleave(8),
            odd_grid,
            forward_map,
        )
        candidate_levels = repaired.reshape(-1, 32)
        quant_levels = (2 * candidate_levels + 1).to(torch.float32)
        sumqx = torch.sum(weight * xval * quant_levels, dim=1)
        sumq2 = torch.sum(weight * quant_levels * quant_levels, dim=1)
        update = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        fitted_scale = sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2))
        scale = torch.where(update, fitted_scale, scale)
        best = torch.where(update, fitted_scale * sumqx, best)
        levels = torch.where(update[:, None], candidate_levels, levels)
        selected_on_grid = torch.where(
            update[:, None], on_grid.reshape(-1, 8), selected_on_grid
        )
    needs_repair = (~selected_on_grid) & (scale[:, None] > 0)
    repair_positions = torch.nonzero(needs_repair.reshape(-1), as_tuple=False).flatten()
    if repair_positions.numel():
        inverse_scale = 1 / torch.where(scale > 0, scale, torch.ones_like(scale))
        rerun_levels = _nearest_int(
            0.5 * (inverse_scale[:, None] * xval - 1)
        ).clamp_(0, K_MAX_Q - 1)
        repaired, _ = _repair_lanes(
            rerun_levels.reshape(-1, 4)[repair_positions],
            x_lanes[repair_positions],
            lane_weight[repair_positions],
            scale.repeat_interleave(8)[repair_positions],
            odd_grid,
            forward_map,
        )
        lane_levels = levels.reshape(-1, 4)
        lane_levels[repair_positions] = repaired
        quant_levels = (2 * levels + 1).to(torch.float32)
        sumqx = torch.sum(weight * xval * quant_levels, dim=1)
        sumq2 = torch.sum(weight * quant_levels * quant_levels, dim=1)
        refitted = sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2))
        scale = torch.where(needs_repair.any(dim=1) & (sumq2 > 0), refitted, scale)
    negative_scale = scale < 0
    scale = scale.abs()
    sign_codes = torch.where(
        negative_scale[:, None], torch.bitwise_not(sign_codes) & 127, sign_codes
    )
    final_indices = forward_map[_packed_levels(levels.reshape(-1, 4))]
    if torch.any(final_indices < 0):
        raise RuntimeError("IQ3_XXS repair emitted an off-grid lane")
    return scale, levels, sign_codes


def _decode_signs(sign_codes: torch.Tensor) -> torch.Tensor:
    bits = torch.arange(8, dtype=torch.int64, device=sign_codes.device)
    lower_bits = bits[:7]
    parity = torch.sum((sign_codes[:, :, None] >> lower_bits) & 1, dim=2) & 1
    decoded = sign_codes | (parity << 7)
    negative = ((decoded[:, :, None] >> bits) & 1).to(torch.bool)
    return torch.where(negative, -1.0, 1.0).reshape(-1, 32)


def _quantize_chunk(
    blocks: torch.Tensor, quant_weights: torch.Tensor
) -> torch.Tensor:
    raw_grid, odd_grid, forward_map = _tables(str(blocks.device))
    sigma2 = 2 * torch.sum(blocks * blocks, dim=1) / QK_K
    subblocks = blocks.reshape(-1, 32)
    subblock_weights = quant_weights.reshape(-1, 32)
    repeated_sigma2 = sigma2.repeat_interleave(8)
    scales, levels, sign_codes = _search_subblocks(
        subblocks, subblock_weights, repeated_sigma2, odd_grid, forward_map
    )
    block_scales = scales.reshape(-1, 8)
    maximum = torch.amax(block_scales, dim=1)
    base_scale = maximum / 31
    stored_base_scale = (base_scale * 1.0125).to(torch.float16).to(torch.float32)
    inverse_base_scale = 1 / torch.where(
        base_scale > 0, base_scale, torch.ones_like(base_scale)
    )
    scale_levels = _nearest_int(
        0.5 * (inverse_base_scale[:, None] * block_scales - 1)
    ).clamp_(0, 15)
    decoded_scale = (
        stored_base_scale[:, None]
        * (0.5 + scale_levels.to(torch.float32))
        * 0.5
    ).reshape(-1, 1)
    lane_indices = forward_map[_packed_levels(levels.reshape(-1, 4))]
    magnitudes = raw_grid[lane_indices].reshape(-1, 32)
    reconstruction = decoded_scale * magnitudes * _decode_signs(sign_codes)
    return reconstruction.reshape(-1, QK_K)


def quantize_rows_reference_torch(
    matrix: torch.Tensor,
    importance: torch.Tensor,
    device: torch.device | str | None = None,
    *,
    chunk: int = 1024,
) -> tuple[torch.Tensor, float]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    target = matrix.device if device is None else torch.device(device)
    work = matrix.to(device=target, dtype=torch.float32)
    channel_weights = importance.to(device=target, dtype=torch.float32).reshape(-1)
    rows, width = work.shape
    usable = width // QK_K * QK_K
    if channel_weights.numel() < usable:
        raise ValueError("importance is shorter than the quantized width")
    output = work.clone()
    if usable == 0:
        return output.to(dtype=matrix.dtype), 0.0
    blocks_per_row = usable // QK_K
    source_blocks = work[:, :usable].contiguous().reshape(-1, QK_K)
    output_blocks = torch.empty_like(source_blocks)
    importance_blocks = channel_weights[:usable].reshape(-1, QK_K)
    error = torch.zeros((), dtype=torch.float64, device=target)
    energy = torch.zeros((), dtype=torch.float64, device=target)
    for start in range(0, len(source_blocks), chunk):
        stop = min(start + chunk, len(source_blocks))
        block_indices = torch.arange(start, stop, device=target) % blocks_per_row
        block_weights = importance_blocks[block_indices]
        reconstruction = _quantize_chunk(source_blocks[start:stop], block_weights)
        output_blocks[start:stop] = reconstruction
        differences = source_blocks[start:stop] - reconstruction
        error += torch.sum((block_weights * differences * differences).to(torch.float64))
        energy += torch.sum(
            (block_weights * source_blocks[start:stop] * source_blocks[start:stop]).to(
                torch.float64
            )
        )
    output[:, :usable] = output_blocks.reshape(rows, usable)
    relative_error = float((error / torch.clamp(energy, min=1e-30)).item())
    return output.to(dtype=matrix.dtype), relative_error
