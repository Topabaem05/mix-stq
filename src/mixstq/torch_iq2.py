from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

QK_BLOCK = 256
LANE_CHUNK = 1048576

TIERS = {
    "iq2_xxs": {"table": "iq2xxs_grid", "lane": 8, "bpw": 2.0625},
    "iq2_xs": {"table": "iq2xs_grid", "lane": 8, "bpw": 2.3125},
    "iq2_s": {"table": "iq2s_grid", "lane": 8, "bpw": 2.5625},
    "iq3_xxs": {"table": "iq3xxs_grid", "lane": 4, "bpw": 3.0625},
    "iq3_s": {"table": "iq3s_grid", "lane": 4, "bpw": 3.4375},
}
LANE = TIERS["iq2_xxs"]["lane"]
LANES_PER_BLOCK = QK_BLOCK // LANE
BLOCK_BYTES = 2 + (QK_BLOCK // 8) * 2
IQ2XXS_BPW = TIERS["iq2_xxs"]["bpw"]


@lru_cache(maxsize=8)
def grid(device_str: str, tier: str = "iq2_xxs") -> torch.Tensor:
    table_name = TIERS[tier]["table"]
    if table_name == "iq2xxs_grid":
        payload = json.loads(
            (Path(__file__).with_name("iq2xxs_tables.json")).read_text(encoding="utf-8")
        )
        points = payload["grid"]
    else:
        payload = json.loads(
            (Path(__file__).with_name("tier_tables.json")).read_text(encoding="utf-8")
        )
        points = payload[table_name]
    return torch.tensor(points, dtype=torch.float32, device=torch.device(device_str))


def _solve_chunk(magnitude, weights, table, table_square):
    linear = ((weights * magnitude) @ table.t()).clamp_min(0.0)
    quadratic = (weights @ table_square.t()).clamp_min(1e-30)
    objective = -linear.square() / quadratic
    index = objective.argmin(dim=1)
    selected = index.unsqueeze(1)
    step = (linear.gather(1, selected) / quadratic.gather(1, selected)).squeeze(1)
    return index, step.clamp_min(0.0)


def quantize_rows(matrix, importance, tier="iq2_xxs"):
    device = matrix.device
    lane = TIERS[tier]["lane"]
    lanes_per_block = QK_BLOCK // lane
    table = grid(str(device), tier)
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
    block_step = max(LANE_CHUNK // lanes_per_block, 1)
    for start in range(0, body.shape[0], block_step):
        block = body[start : start + block_step]
        block_weight = weight_body[start : start + block_step]
        lanes = block.reshape(-1, lane)
        lane_weights = block_weight.reshape(-1, lane)
        signs = torch.where(lanes < 0, -1.0, 1.0)
        magnitude = lanes.abs()
        index, step = _solve_chunk(magnitude, lane_weights, table, table_square)
        recon[start : start + block_step] = (
            table[index] * step.unsqueeze(1) * signs
        ).reshape(block.shape)

    total_error = (weight_body * (body - recon).square()).double().sum()
    total_energy = (weight_body * body.square()).double().sum()
    out = work.clone()
    out[:, :usable] = recon.reshape(rows, usable)
    return out.to(matrix.dtype), float(total_error / total_energy.clamp_min(1e-30))
