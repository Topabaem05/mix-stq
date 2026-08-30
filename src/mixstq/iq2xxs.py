from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

QK_K = 256
SUBBLOCK = 32
LANE = 8
BLOCK_BYTES = 2 + (QK_K // 8) * 2
IQ2XXS_BPW = BLOCK_BYTES * 8.0 / QK_K


@lru_cache(maxsize=1)
def tables() -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((Path(__file__).with_name("iq2xxs_tables.json")).read_text(encoding="utf-8"))
    grid = np.asarray(payload["grid"], dtype=np.float64)
    kmask = np.asarray(payload["kmask"], dtype=np.uint8)
    ksigns = np.asarray(payload["ksigns"], dtype=np.uint8)
    sign_patterns = np.where(
        (ksigns[:, None] & kmask[None, :]) != 0, -1.0, 1.0
    )
    return grid, sign_patterns


def _best_lane_codes(lane_values, grid, sign_patterns, step):
    magnitudes = np.abs(lane_values)
    signs = np.sign(lane_values)
    signs[signs == 0] = 1.0
    candidates = grid[None, :, :] * step
    cost = np.sum((magnitudes[:, None, :] - candidates) ** 2, axis=2)
    grid_index = np.argmin(cost, axis=1)
    negative = signs < 0
    sign_index = np.zeros(lane_values.shape[0], dtype=np.int64)
    for j in range(LANE):
        sign_index |= (negative[:, j].astype(np.int64) << j)
    parity_valid = sign_index < sign_patterns.shape[0]
    sign_index = np.where(parity_valid, sign_index, sign_index & (sign_patterns.shape[0] - 1))
    return grid_index, sign_index


def encode_block(values, importance):
    grid, _ = tables()
    block = values.reshape(-1, LANE)
    weights = importance.reshape(-1, LANE)
    magnitude_scale = float(np.max(np.abs(values))) / 43.0
    if magnitude_scale <= 0.0:
        return np.zeros_like(values), 0.0

    best_error = None
    best_recon = None
    for coarse in np.linspace(0.6, 1.4, 9):
        d = magnitude_scale * coarse
        recon = np.zeros_like(block)
        for sub in range(0, block.shape[0], SUBBLOCK // LANE):
            lanes = block[sub : sub + SUBBLOCK // LANE]
            lane_weights = weights[sub : sub + SUBBLOCK // LANE]
            if lanes.size == 0:
                continue
            sub_error = None
            sub_recon = None
            for level in range(16):
                step = d * (0.5 + level) * 0.25
                if step <= 0.0:
                    continue
                magnitudes = np.abs(lanes)
                candidates = grid[None, :, :] * step
                cost = np.sum(
                    lane_weights[:, None, :] * (magnitudes[:, None, :] - candidates) ** 2, axis=2
                )
                index = np.argmin(cost, axis=1)
                signs = np.where(lanes < 0, -1.0, 1.0)
                trial = grid[index] * step * signs
                error = float(np.sum(lane_weights * (lanes - trial) ** 2))
                if sub_error is None or error < sub_error:
                    sub_error = error
                    sub_recon = trial
            recon[sub : sub + SUBBLOCK // LANE] = sub_recon
        total = float(np.sum(weights * (block - recon) ** 2))
        if best_error is None or total < best_error:
            best_error = total
            best_recon = recon.copy()
    return best_recon.reshape(-1), best_error


def encode(values, importance):
    usable = (values.size // QK_K) * QK_K
    if usable == 0:
        raise ValueError("need at least one 256-weight block")
    total_error = 0.0
    energy = 0.0
    for start in range(0, usable, QK_K):
        chunk = values[start : start + QK_K]
        weight_chunk = importance[start : start + QK_K]
        _, error = encode_block(chunk, weight_chunk)
        total_error += error
        energy += float(np.sum(weight_chunk * chunk ** 2))
    return {
        "relative_error": total_error / max(energy, 1e-30),
        "bpw": IQ2XXS_BPW,
        "blocks": usable // QK_K,
    }
