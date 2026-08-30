from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    group: str
    tier: str
    error: float
    bytes_cost: int


def lagrangian_allocate(candidates, budget_bytes, iterations=60):
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.group, []).append(candidate)

    def solve(multiplier):
        chosen = {}
        for name, options in groups.items():
            chosen[name] = min(options, key=lambda c: c.error + multiplier * c.bytes_cost)
        return chosen

    low, high = 0.0, 1.0
    while sum(c.bytes_cost for c in solve(high).values()) > budget_bytes:
        high *= 2.0
        if high > 1e12:
            break
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if sum(c.bytes_cost for c in solve(mid).values()) > budget_bytes:
            low = mid
        else:
            high = mid
    chosen = solve(high)
    return chosen, sum(c.bytes_cost for c in chosen.values()), sum(c.error for c in chosen.values())


def topk_marginal_allocate(candidates, budget_bytes):
    groups: dict[str, dict[str, Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.group, {})[candidate.tier] = candidate
    cost_by_tier: dict[str, int] = {}
    for candidate in candidates:
        cost_by_tier[candidate.tier] = candidate.bytes_cost
    tiers = sorted(cost_by_tier, key=lambda tier: cost_by_tier[tier])
    base_tier, up_tier = tiers[0], tiers[-1]
    chosen = {name: options[base_tier] for name, options in groups.items()}
    used = sum(c.bytes_cost for c in chosen.values())
    ranked = sorted(
        groups.items(),
        key=lambda item: (item[1][base_tier].error - item[1][up_tier].error)
        / max(item[1][up_tier].bytes_cost - item[1][base_tier].bytes_cost, 1),
        reverse=True,
    )
    for name, options in ranked:
        delta = options[up_tier].bytes_cost - options[base_tier].bytes_cost
        if used + delta <= budget_bytes:
            chosen[name] = options[up_tier]
            used += delta
    return chosen, used, sum(c.error for c in chosen.values())
