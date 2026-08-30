from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

QK_BLOCK = 256
LANE = 8
LANES_PER_BLOCK = QK_BLOCK // LANE
BLOCK_BYTES = 2 + (QK_BLOCK // 8) * 2
IQ2XXS_BPW = BLOCK_BYTES * 8.0 / QK_BLOCK
SUB_LEVELS = 16
COARSE = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)
LANE_CHUNK = 1048576


@lru_cache(maxsize=4)
def grid(device_str: str) -> torch.Tensor:
    payload = json.loads(
        (Path(__file__).with_name("iq2xxs_tables.json")).read_text(encoding="utf-8")
    )
    return torch.tensor(payload["grid"], dtype=torch.float32, device=torch.device(device_str))


def _solve_chunk(magnitude, weights, base, table, table_square):
    linear = (weights * magnitude) @ table.t()
    quadratic = weights @ table_square.t()
    lane_base = base.repeat_interleave(LANES_PER_BLOCK, dim=0)

    best_cost = None
    best_index = None
    best_step = None
    for coarse in COARSE:
        scaled = lane_base * coarse
        for level in range(SUB_LEVELS):
            step = scaled * ((level + 0.5) * 0.25)
            objective = quadratic * step.square() - 2.0 * linear * step
            index = objective.argmin(dim=1)
            cost = objective.gather(1, index.unsqueeze(1)).squeeze(1)
            if best_cost is None:
                best_cost = cost
                best_index = index
                best_step = step.squeeze(1)
            else:
                improved = cost < best_cost
                best_cost = torch.where(improved, cost, best_cost)
                best_index = torch.where(improved, index, best_index)
                best_step = torch.where(improved, step.squeeze(1), best_step)
    return best_index, best_step


def quantize_rows(matrix, importance):
    device = matrix.device
    table = grid(str(device))
    table_square = table.square()
    work = matrix.to(torch.float32)
    rows, width = work.shape
    usable = (width // QK_BLOCK) * QK_BLOCK
    if usable == 0:
        return matrix.clone(), 0.0

    body = work[:, :usable].reshape(-1, QK_BLOCK)
    channel = importance.to(torch.float32)[:usable]
    weight_body = channel.reshape(1, usable).expand(rows, usable).reshape(-1, QK_BLOCK)

    recon = torch.empty_like(body)
    block_step = max(LANE_CHUNK // LANES_PER_BLOCK, 1)
    for start in range(0, body.shape[0], block_step):
        block = body[start : start + block_step]
        block_weight = weight_body[start : start + block_step]
        base = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 43.0
        lanes = block.reshape(-1, LANE)
        lane_weights = block_weight.reshape(-1, LANE)
        signs = torch.where(lanes < 0, -1.0, 1.0)
        magnitude = lanes.abs()
        index, step = _solve_chunk(magnitude, lane_weights, base, table, table_square)
        recon[start : start + block_step] = (
            table[index] * step.unsqueeze(1) * signs
        ).reshape(block.shape)

    total_error = (weight_body * (body - recon).square()).double().sum()
    total_energy = (weight_body * body.square()).double().sum()
    out = work.clone()
    out[:, :usable] = recon.reshape(rows, usable)
    return out.to(matrix.dtype), float(total_error / total_energy.clamp_min(1e-30))

