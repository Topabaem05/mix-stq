from __future__ import annotations

import itertools

import numpy as np

IQ2_XXS_BLOCK = 256
IQ2_XXS_BYTES_PER_BLOCK = 66
LANES = 8
CODEBOOK_BITS = 8


def iq2_xxs_bits_per_weight() -> float:
    return IQ2_XXS_BYTES_PER_BLOCK * 8.0 / IQ2_XXS_BLOCK


def build_grid(size: int = 2 ** CODEBOOK_BITS) -> np.ndarray:
    magnitudes = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    space = np.array(list(itertools.product(magnitudes, repeat=4)), dtype=np.float32)
    energy = space.sum(axis=1)
    keep = space[np.argsort(energy)][:32]
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=4)), dtype=np.float32)
    rows = []
    for base in keep:
        for sign in signs:
            rows.append(base * sign)
    grid = np.unique(np.array(rows, dtype=np.float32), axis=0)
    return grid[:size]


def encode_iq2_xxs(values: np.ndarray, importance: np.ndarray, rounds: int = 3) -> dict:
    usable = (values.size // 4) * 4
    groups = values[:usable].reshape(-1, 4).astype(np.float32)
    weights = importance[:usable].reshape(-1, 4).astype(np.float32)
    grid = build_grid()
    scale = float(
        np.sum(weights * np.abs(groups)) / max(np.sum(weights * np.ones_like(groups)), 1e-12)
    )
    numerator = 0.0
    denominator = 0.0
    total = 0.0
    for _ in range(rounds):
        numerator = 0.0
        denominator = 0.0
        total = 0.0
        for start in range(0, groups.shape[0], 16384):
            g = groups[start : start + 16384]
            w = weights[start : start + 16384]
            residual = g[:, None, :] - scale * grid[None, :, :]
            cost = np.sum(w[:, None, :] * residual * residual, axis=2)
            pick = np.argmin(cost, axis=1)
            selected = grid[pick]
            numerator += float(np.sum(w * g * selected))
            denominator += float(np.sum(w * selected * selected))
            total += float(cost[np.arange(g.shape[0]), pick].sum())
        if denominator > 0:
            scale = numerator / denominator
    energy = float(np.sum(weights * groups * groups))
    return {
        "bpw": iq2_xxs_bits_per_weight(),
        "grid_entries": int(grid.shape[0]),
        "relative_error": total / max(energy, 1e-30),
        "scale": scale,
    }
