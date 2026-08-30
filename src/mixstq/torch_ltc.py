from __future__ import annotations

import itertools

import torch

LANES = 4
QK_BLOCK = 256
CODEBOOK_SIZE = 32
FIT_SAMPLE_GROUPS = 65536
ASSIGN_CHUNK = 4194304


def all_patterns(device, dtype=torch.float32):
    grid = list(itertools.product([-1.0, 0.0, 1.0], repeat=LANES))
    return torch.tensor(grid, device=device, dtype=dtype)


def stq_patterns(device, dtype=torch.float32):
    every = all_patterns(device, dtype)
    return every[(every == 0).sum(dim=1) == 1]


def _assign(groups, weights, patterns, scale):
    residual = groups.unsqueeze(1) - scale * patterns.unsqueeze(0)
    cost = (weights.unsqueeze(1) * residual.square()).sum(dim=2)
    return cost.argmin(dim=1)


def fit_codebook(groups, weights, scale, rounds=3):
    device = groups.device
    every = all_patterns(device, groups.dtype)
    count = groups.shape[0]
    if count > FIT_SAMPLE_GROUPS:
        stride = count // FIT_SAMPLE_GROUPS + 1
        sample_groups = groups[::stride]
        sample_weights = weights[::stride]
    else:
        sample_groups = groups
        sample_weights = weights

    patterns = stq_patterns(device, groups.dtype)
    for _ in range(rounds):
        residual = sample_groups.unsqueeze(1) - scale * every.unsqueeze(0)
        cost = (sample_weights.unsqueeze(1) * residual.square()).sum(dim=2)
        best = torch.full((cost.shape[0],), float("inf"), device=device, dtype=cost.dtype)
        chosen: list[int] = []
        for _ in range(CODEBOOK_SIZE):
            gain = torch.clamp(best.unsqueeze(1) - cost, min=0.0).sum(dim=0)
            if chosen:
                gain[torch.tensor(chosen, device=device)] = -1.0
            pick = int(gain.argmax())
            chosen.append(pick)
            best = torch.minimum(best, cost[:, pick])
        patterns = every[torch.tensor(sorted(chosen), device=device)]
        selected = patterns[_assign(sample_groups, sample_weights, patterns, scale)]
        numerator = (sample_weights * sample_groups * selected).sum()
        denominator = (sample_weights * selected.square()).sum()
        if float(denominator) > 0.0:
            scale = float(numerator / denominator)
    return patterns, scale


def quantize_rows(matrix, importance, patterns=None, learn=True, rounds=3):
    device = matrix.device
    work = matrix.to(torch.float32)
    rows, width = work.shape
    usable = (width // QK_BLOCK) * QK_BLOCK
    if usable == 0:
        return matrix.clone(), 0.0

    body = work[:, :usable].reshape(-1, QK_BLOCK)
    channel = importance.to(torch.float32)[:usable]
    weight_body = channel.reshape(1, usable).expand(rows, usable).reshape(-1, QK_BLOCK)

    groups = body.reshape(-1, LANES)
    weights = weight_body.reshape(-1, LANES)
    scale_guess = float((weights * groups.abs()).sum() / weights.sum().clamp_min(1e-12))

    if learn:
        patterns, _ = fit_codebook(groups, weights, scale_guess, rounds=rounds)
    if patterns is None:
        patterns = stq_patterns(device, torch.float32)

    block_groups = QK_BLOCK // LANES
    recon = torch.empty_like(body)
    total_error = torch.zeros((), device=device, dtype=torch.float64)
    total_energy = torch.zeros((), device=device, dtype=torch.float64)

    step = max(ASSIGN_CHUNK // QK_BLOCK, 1)
    for start in range(0, body.shape[0], step):
        block = body[start : start + step]
        block_weight = weight_body[start : start + step]
        g = block.reshape(-1, LANES)
        w = block_weight.reshape(-1, LANES)
        scale = torch.full((block.shape[0],), scale_guess, device=device, dtype=torch.float32)
        for _ in range(rounds):
            expanded = scale.repeat_interleave(block_groups).unsqueeze(1).unsqueeze(2)
            residual = g.unsqueeze(1) - expanded * patterns.unsqueeze(0)
            cost = (w.unsqueeze(1) * residual.square()).sum(dim=2)
            selected = patterns[cost.argmin(dim=1)]
            sel = selected.reshape(block.shape[0], QK_BLOCK)
            numerator = (block_weight * block * sel).sum(dim=1)
            denominator = (block_weight * sel.square()).sum(dim=1)
            scale = torch.where(denominator > 0, numerator / denominator, scale)
        scale = scale.to(torch.float16).to(torch.float32)
        expanded = scale.repeat_interleave(block_groups).unsqueeze(1).unsqueeze(2)
        residual = g.unsqueeze(1) - expanded * patterns.unsqueeze(0)
        cost = (w.unsqueeze(1) * residual.square()).sum(dim=2)
        selected = patterns[cost.argmin(dim=1)].reshape(block.shape[0], QK_BLOCK)
        approx = selected * scale.unsqueeze(1)
        recon[start : start + step] = approx
        total_error += (block_weight * (block - approx).square()).sum().double()
        total_energy += (block_weight * block.square()).sum().double()

    out = work.clone()
    out[:, :usable] = recon.reshape(rows, usable)
    relative = float(total_error / total_energy.clamp_min(1e-30))
    return out.to(matrix.dtype), relative
