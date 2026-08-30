from __future__ import annotations

import itertools

import numpy as np

LANES = 4
BLOCK = 256
CODE_BITS = 5
CODEBOOK_SIZE = 2 ** CODE_BITS

ALL_PATTERNS = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=LANES)))
STQ_PATTERNS = ALL_PATTERNS[np.sum(ALL_PATTERNS == 0.0, axis=1) == 1]


def assign(groups, weights, patterns, scale):
    residual = groups[:, None, :] - scale * patterns[None, :, :]
    cost = np.sum(weights[:, None, :] * residual ** 2, axis=2)
    choice = np.argmin(cost, axis=1)
    return choice, cost[np.arange(groups.shape[0]), choice]


def refit_scale(groups, weights, patterns, choice):
    selected = patterns[choice]
    numerator = float(np.sum(weights * groups * selected))
    denominator = float(np.sum(weights * selected ** 2))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def encode_fixed(values, importance, patterns, rounds=3):
    groups = values.reshape(-1, LANES)
    weights = importance.reshape(-1, LANES)
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    choice = None
    for _ in range(rounds):
        choice, _ = assign(groups, weights, patterns, scale)
        scale = refit_scale(groups, weights, patterns, choice)
    choice, errors = assign(groups, weights, patterns, scale)
    return scale, choice, float(np.sum(errors))


def select_codebook(groups, weights, scale, size=CODEBOOK_SIZE):
    residual = groups[:, None, :] - scale * ALL_PATTERNS[None, :, :]
    cost = np.sum(weights[:, None, :] * residual ** 2, axis=2)
    chosen: list[int] = []
    best = np.full(groups.shape[0], np.inf)
    for _ in range(size):
        gain = np.maximum(best[:, None] - cost, 0.0).sum(axis=0)
        gain[chosen] = -1.0
        pick = int(np.argmax(gain))
        chosen.append(pick)
        best = np.minimum(best, cost[:, pick])
    return ALL_PATTERNS[np.array(sorted(chosen))]


def encode_learned(values, importance, rounds=3):
    groups = values.reshape(-1, LANES)
    weights = importance.reshape(-1, LANES)
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    patterns = STQ_PATTERNS
    for _ in range(rounds):
        patterns = select_codebook(groups, weights, scale)
        choice, _ = assign(groups, weights, patterns, scale)
        scale = refit_scale(groups, weights, patterns, choice)
    choice, errors = assign(groups, weights, patterns, scale)
    return scale, patterns, choice, float(np.sum(errors))

