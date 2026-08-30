from __future__ import annotations

import numpy as np
from frontier import _cost, evaluate, pattern_space


def seed_by_contribution(groups, weights, scale, space, size, sample=4096):
    if groups.shape[0] > sample:
        stride = groups.shape[0] // sample + 1
        probe_groups = groups[::stride]
        probe_weights = weights[::stride]
    else:
        probe_groups = groups
        probe_weights = weights
    cost = _cost(probe_groups, probe_weights, space, scale)
    best = np.argmin(cost, axis=1)
    zero_index = int(np.argmin(np.sum(np.abs(space), axis=1)))
    reduction = np.maximum(cost[:, zero_index] - cost[np.arange(cost.shape[0]), best], 0.0)
    scores = np.bincount(best, weights=reduction, minlength=space.shape[0])
    order = np.argsort(-scores)[:size]
    return np.sort(order), cost


def local_exchange(cost, chosen, space_size, sweeps=4):
    chosen = list(chosen)
    member = np.zeros(space_size, dtype=bool)
    member[chosen] = True
    for _ in range(sweeps):
        improved = False
        active = np.array(chosen)
        sub = cost[:, active]
        rows = np.arange(sub.shape[0])
        order = np.argsort(sub, axis=1)
        best_idx = active[order[:, 0]]
        best_val = sub[rows, order[:, 0]]
        second_val = sub[rows, order[:, 1]] if sub.shape[1] > 1 else best_val
        loss_if_dropped = np.zeros(space_size)
        np.add.at(loss_if_dropped, best_idx, second_val - best_val)
        candidates = np.where(~member)[0]
        if candidates.size == 0:
            break
        gain = np.maximum(best_val[:, None] - cost[:, candidates], 0.0).sum(axis=0)
        add_pick = int(candidates[int(np.argmax(gain))])
        add_gain = float(gain.max())
        drop_costs = loss_if_dropped[active]
        drop_pos = int(np.argmin(drop_costs))
        drop_pick = int(active[drop_pos])
        drop_cost = float(drop_costs[drop_pos])
        if add_gain > drop_cost + 1e-12 and add_pick != drop_pick:
            member[drop_pick] = False
            member[add_pick] = True
            chosen.remove(drop_pick)
            chosen.append(add_pick)
            improved = True
        if not improved:
            break
    return np.sort(np.array(chosen))


def encode_hybrid(values, importance, lanes, code_bits, rounds=3, sweeps=4):
    usable = (values.size // lanes) * lanes
    groups = values[:usable].reshape(-1, lanes).astype(np.float32)
    weights = importance[:usable].reshape(-1, lanes).astype(np.float32)
    space = pattern_space(lanes)
    size = min(2 ** code_bits, space.shape[0])
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    patterns = None
    for _ in range(rounds):
        chosen, cost = seed_by_contribution(groups, weights, scale, space, size)
        refined = local_exchange(cost, chosen, space.shape[0], sweeps)
        patterns = space[refined]
        _, scale, _ = evaluate(groups, weights, patterns, scale)
    total, _, zero_rate = evaluate(groups, weights, patterns, scale)
    energy = float(np.sum(weights * groups * groups))
    return {
        "lanes": lanes,
        "code_bits": code_bits,
        "codebook_entries": int(patterns.shape[0]),
        "relative_error": total / max(energy, 1e-30),
        "zero_rate": zero_rate,
    }
