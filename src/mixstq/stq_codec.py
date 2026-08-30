from __future__ import annotations

import numpy as np

BLOCK = 256
LANES = 4
GROUPS_PER_BLOCK = BLOCK // LANES
STQ_BYTES_PER_BLOCK = GROUPS_PER_BLOCK * 5 // 8 + 2


def stq_bits_per_weight() -> float:
    return STQ_BYTES_PER_BLOCK * 8.0 / BLOCK


def _weighted_error(values, importance, scale, signs):
    recon = scale * signs
    return float(np.sum(importance * (values - recon) ** 2))


def _refit_scale(values, importance, signs):
    active = signs != 0
    numerator = float(np.sum(importance[active] * np.abs(values[active])))
    denominator = float(np.sum(importance[active]))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _stq_signs(values, importance, scale):
    groups = values.reshape(-1, LANES)
    weights = importance.reshape(-1, LANES)
    signs = np.sign(groups)
    signs[signs == 0] = 1.0
    marginal = weights * (2.0 * np.abs(groups) - scale)
    zero_lane = np.argmin(marginal, axis=1)
    signs[np.arange(groups.shape[0]), zero_lane] = 0.0
    return signs.reshape(-1)


def _free_ternary_signs(values, importance, scale):
    signs = np.sign(values)
    signs[signs == 0] = 1.0
    marginal = importance * (2.0 * np.abs(values) - scale)
    signs[marginal >= 0.0] = signs[marginal >= 0.0]
    signs[marginal < 0.0] = 0.0
    return signs


def encode_stq_block(values, importance, rounds=3):
    scale = _refit_scale(values, importance, np.ones_like(values))
    trace = []
    signs = None
    for _ in range(rounds):
        signs = _stq_signs(values, importance, scale)
        trace.append(_weighted_error(values, importance, scale, signs))
        scale = _refit_scale(values, importance, signs)
        trace.append(_weighted_error(values, importance, scale, signs))
    return scale, signs, trace


def encode_free_ternary_block(values, importance, rounds=3):
    scale = _refit_scale(values, importance, np.ones_like(values))
    signs = None
    for _ in range(rounds):
        signs = _free_ternary_signs(values, importance, scale)
        scale = _refit_scale(values, importance, signs)
    return scale, signs


def sparsity_headroom(values, importance, rounds=3):
    scale_s, signs_s, _ = encode_stq_block(values, importance, rounds)
    scale_t, signs_t = encode_free_ternary_block(values, importance, rounds)
    error_stq = _weighted_error(values, importance, scale_s, signs_s)
    error_tern = _weighted_error(values, importance, scale_t, signs_t)
    energy = float(np.sum(importance * values ** 2))
    natural_zero_fraction = float(np.mean(signs_t == 0.0))
    return {
        "error_stq": error_stq,
        "error_free_ternary": error_tern,
        "energy": energy,
        "relative_stq": error_stq / energy,
        "relative_free_ternary": error_tern / energy,
        "structural_penalty": (error_stq - error_tern) / energy,
        "natural_zero_fraction": natural_zero_fraction,
        "exactly_one_zero_per_group": bool(
            np.all(np.sum(signs_s.reshape(-1, LANES) == 0.0, axis=1) == 1)
        ),
    }

