from __future__ import annotations

import numpy as np
from codebook import ALL_PATTERNS, CODEBOOK_SIZE, LANES


def group_histogram(groups, weights, scale, sample_cap=200000):
    count = groups.shape[0]
    if count > sample_cap:
        stride = count // sample_cap + 1
        groups = groups[::stride]
        weights = weights[::stride]
    residual = groups[:, None, :] - scale * ALL_PATTERNS[None, :, :]
    return np.sum(weights[:, None, :] * residual ** 2, axis=2)


def select_codebook_fast(groups, weights, scale, size=CODEBOOK_SIZE):
    cost = group_histogram(groups, weights, scale)
    chosen: list[int] = []
    best = np.full(cost.shape[0], np.inf)
    for _ in range(size):
        gain = np.maximum(best[:, None] - cost, 0.0).sum(axis=0)
        if chosen:
            gain[np.array(chosen)] = -1.0
        pick = int(np.argmax(gain))
        chosen.append(pick)
        best = np.minimum(best, cost[:, pick])
    return ALL_PATTERNS[np.array(sorted(chosen))]


def assign_chunked(groups, weights, patterns, scale, chunk=200000):
    total_error = 0.0
    zero_hits = 0
    elements = 0
    for start in range(0, groups.shape[0], chunk):
        g = groups[start : start + chunk]
        w = weights[start : start + chunk]
        residual = g[:, None, :] - scale * patterns[None, :, :]
        cost = np.sum(w[:, None, :] * residual ** 2, axis=2)
        choice = np.argmin(cost, axis=1)
        total_error += float(cost[np.arange(g.shape[0]), choice].sum())
        selected = patterns[choice]
        zero_hits += int(np.sum(selected == 0.0))
        elements += selected.size
    return total_error, zero_hits / max(elements, 1)


def refit_scale_chunked(groups, weights, patterns, scale, chunk=200000):
    numerator = 0.0
    denominator = 0.0
    for start in range(0, groups.shape[0], chunk):
        g = groups[start : start + chunk]
        w = weights[start : start + chunk]
        residual = g[:, None, :] - scale * patterns[None, :, :]
        cost = np.sum(w[:, None, :] * residual ** 2, axis=2)
        selected = patterns[np.argmin(cost, axis=1)]
        numerator += float(np.sum(w * g * selected))
        denominator += float(np.sum(w * selected ** 2))
    if denominator <= 0.0:
        return scale
    return numerator / denominator


def encode(values, importance, patterns=None, rounds=3, learn=True):
    groups = values.reshape(-1, LANES)
    weights = importance.reshape(-1, LANES)
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    for _ in range(rounds):
        if learn:
            patterns = select_codebook_fast(groups, weights, scale)
        scale = refit_scale_chunked(groups, weights, patterns, scale)
    error, zero_rate = assign_chunked(groups, weights, patterns, scale)
    energy = float(np.sum(weights * groups ** 2))
    return {
        "scale": scale,
        "patterns": patterns,
        "relative_error": error / max(energy, 1e-30),
        "zero_rate": zero_rate,
    }

