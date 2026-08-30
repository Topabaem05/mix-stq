from __future__ import annotations

import torch
import torch_iq2 as tq

failures = []


def swept_relative_error(matrix, importance, tier):
    lane = tq.TIERS[tier]["lane"]
    lanes_per_block = tq.QK_BLOCK // lane
    table = tq.grid(str(matrix.device), tier)
    table_square = table.square()
    body = matrix.to(torch.float32).reshape(-1, tq.QK_BLOCK)
    weights = importance.to(torch.float32).reshape(1, -1).expand(matrix.shape[0], -1)
    weight_body = weights.reshape(-1, tq.QK_BLOCK)
    lane_weights = weight_body.reshape(-1, lane)
    lanes = body.reshape(-1, lane)
    magnitude = lanes.abs()
    signs = torch.where(lanes < 0, -1.0, 1.0)
    linear = (lane_weights * magnitude) @ table.t()
    quadratic = lane_weights @ table_square.t()
    block_base = body.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / table.max()
    lane_base = block_base.repeat_interleave(lanes_per_block, dim=0).squeeze(1)
    best_cost = torch.full((lanes.shape[0],), torch.inf)
    best_index = torch.zeros(lanes.shape[0], dtype=torch.long)
    best_step = torch.zeros(lanes.shape[0])
    for multiplier in torch.linspace(0.0, 5.25, 22):
        step = lane_base * multiplier
        objective = quadratic * step.unsqueeze(1).square() - 2.0 * linear * step.unsqueeze(1)
        index = objective.argmin(dim=1)
        cost = objective.gather(1, index.unsqueeze(1)).squeeze(1)
        improved = cost < best_cost
        best_cost = torch.where(improved, cost, best_cost)
        best_index = torch.where(improved, index, best_index)
        best_step = torch.where(improved, step, best_step)
    recon = (table[best_index] * best_step.unsqueeze(1) * signs).reshape(body.shape)
    total_error = (weight_body * (body - recon).square()).double().sum()
    total_energy = (weight_body * body.square()).double().sum()
    return float(total_error / total_energy.clamp_min(1e-30))


torch.manual_seed(0)
matrix = torch.randn(8, 256)
importance = torch.rand(256) + 0.1
errors = {}
for tier in ("iq2_xxs", "iq2_s", "iq3_xxs"):
    quantized, analytic_error = tq.quantize_rows(matrix, importance, tier=tier)
    sweep_error = swept_relative_error(matrix, importance, tier)
    errors[tier] = analytic_error
    if quantized.shape != matrix.shape:
        failures.append("tier %s changed shape" % tier)
    if not (0.0 < analytic_error < 1.0):
        failures.append("tier %s relative error %.9f out of range" % (tier, analytic_error))
    if analytic_error > sweep_error + 1e-9:
        failures.append(
            "tier %s analytic error %.9f exceeds sweep error %.9f"
            % (tier, analytic_error, sweep_error)
        )

if not (errors["iq3_xxs"] < errors["iq2_s"] < errors["iq2_xxs"]):
    failures.append("error must decrease as bits increase: %s" % errors)

if failures:
    for line in failures:
        print("FAIL: " + line)
    raise SystemExit(1)
print("PASS: analytic scales beat explicit sweeps across all tiers")
