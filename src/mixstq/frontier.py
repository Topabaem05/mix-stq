from __future__ import annotations

import itertools

import numpy as np

SCALE_BITS_PER_BLOCK = 16.0
BLOCK = 256


def bits_per_weight(lanes: int, code_bits: int) -> float:
    return code_bits / lanes + SCALE_BITS_PER_BLOCK / BLOCK


def pattern_space(lanes: int) -> np.ndarray:
    return np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=lanes)), dtype=np.float32)


def three_four_patterns() -> np.ndarray:
    space = pattern_space(4)
    return space[np.sum(space == 0.0, axis=1) == 1]


def _cost(groups: np.ndarray, weights: np.ndarray, patterns: np.ndarray, scale: float) -> np.ndarray:
    residual = groups[:, None, :] - scale * patterns[None, :, :]
    return np.sum(weights[:, None, :] * residual * residual, axis=2)


def select_by_contribution(groups, weights, scale, patterns, size, sample=4096):
    if groups.shape[0] > sample:
        stride = groups.shape[0] // sample + 1
        groups = groups[::stride]
        weights = weights[::stride]
    cost = _cost(groups, weights, patterns, scale)
    best = np.argmin(cost, axis=1)
    zero_index = int(np.argmin(np.sum(np.abs(patterns), axis=1)))
    reduction = np.maximum(cost[:, zero_index] - cost[np.arange(cost.shape[0]), best], 0.0)
    scores = np.bincount(best, weights=reduction, minlength=patterns.shape[0])
    chosen = np.argsort(-scores)[:size]
    return patterns[np.sort(chosen)]


def evaluate(groups, weights, patterns, scale, chunk=8192):
    total = 0.0
    numerator = 0.0
    denominator = 0.0
    zeros = 0
    elements = 0
    for start in range(0, groups.shape[0], chunk):
        g = groups[start : start + chunk]
        w = weights[start : start + chunk]
        cost = _cost(g, w, patterns, scale)
        pick = np.argmin(cost, axis=1)
        total += float(cost[np.arange(g.shape[0]), pick].sum())
        selected = patterns[pick]
        numerator += float(np.sum(w * g * selected))
        denominator += float(np.sum(w * selected * selected))
        zeros += int(np.sum(selected == 0.0))
        elements += selected.size
    refit = numerator / denominator if denominator > 0 else scale
    return total, refit, zeros / max(elements, 1)


def encode_config(values, importance, lanes, code_bits, patterns=None, rounds=3):
    usable = (values.size // lanes) * lanes
    groups = values[:usable].reshape(-1, lanes).astype(np.float32)
    weights = importance[:usable].reshape(-1, lanes).astype(np.float32)
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    learn = patterns is None
    space = pattern_space(lanes) if learn else patterns
    size = min(2 ** code_bits, space.shape[0])
    active = space if not learn else None
    for _ in range(rounds):
        if learn:
            active = select_by_contribution(groups, weights, scale, space, size)
        _, scale, _ = evaluate(groups, weights, active, scale)
    total, _, zero_rate = evaluate(groups, weights, active, scale)
    energy = float(np.sum(weights * groups * groups))
    return {
        "lanes": lanes,
        "code_bits": code_bits,
        "bpw": bits_per_weight(lanes, code_bits),
        "codebook_entries": int(active.shape[0]),
        "pattern_space": int(space.shape[0]),
        "relative_error": total / max(energy, 1e-30),
        "zero_rate": zero_rate,
        "scale": scale,
    }

