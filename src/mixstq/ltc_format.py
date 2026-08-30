from __future__ import annotations

import struct

import numpy as np
from codebook import ALL_PATTERNS, CODEBOOK_SIZE, LANES

QK_BLOCK = 256
GROUPS_PER_BLOCK = QK_BLOCK // LANES
CODE_BITS = 5
BLOCK_BYTES = GROUPS_PER_BLOCK * CODE_BITS // 8 + 2
LTC_BPW = BLOCK_BYTES * 8.0 / QK_BLOCK
CODEBOOK_BYTES = CODEBOOK_SIZE


def pattern_to_index(pattern) -> int:
    total = 0
    for lane in range(LANES):
        digit = int(pattern[lane]) + 1
        total += digit * (3 ** lane)
    return total


def index_to_pattern(index: int) -> np.ndarray:
    values = np.zeros(LANES, dtype=np.float64)
    remainder = index
    for lane in range(LANES):
        values[lane] = float(remainder % 3) - 1.0
        remainder //= 3
    return values


def serialize_codebook(patterns) -> bytes:
    if len(patterns) != CODEBOOK_SIZE:
        raise ValueError("codebook must hold exactly %d patterns" % CODEBOOK_SIZE)
    indices = sorted(pattern_to_index(p) for p in patterns)
    if len(set(indices)) != CODEBOOK_SIZE:
        raise ValueError("codebook contains duplicate patterns")
    if any(index > 80 for index in indices):
        raise ValueError("pattern index outside the ternary 4-tuple space")
    return bytes(indices)


def canonical_codebook(patterns) -> np.ndarray:
    indices = sorted(pattern_to_index(p) for p in patterns)
    if len(set(indices)) != CODEBOOK_SIZE:
        raise ValueError("codebook contains duplicate patterns")
    return np.stack([index_to_pattern(index) for index in indices])


def fit_codebook(values, importance, rounds=3):
    from codebook import select_codebook

    groups = values.reshape(-1, LANES)
    weights = importance.reshape(-1, LANES)
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    patterns = stq_codebook()
    for _ in range(rounds):
        patterns = select_codebook(groups, weights, scale)
        residual = groups[:, None, :] - scale * patterns[None, :, :]
        cost = np.sum(weights[:, None, :] * residual ** 2, axis=2)
        selected = patterns[np.argmin(cost, axis=1)]
        numerator = float(np.sum(weights * groups * selected))
        denominator = float(np.sum(weights * selected ** 2))
        if denominator > 0.0:
            scale = numerator / denominator
    return canonical_codebook(patterns)


def deserialize_codebook(payload: bytes) -> np.ndarray:
    if len(payload) != CODEBOOK_BYTES:
        raise ValueError("codebook payload must be %d bytes" % CODEBOOK_BYTES)
    return np.stack([index_to_pattern(byte) for byte in payload])


def pack_codes(code_indices) -> bytes:
    if code_indices.size != GROUPS_PER_BLOCK:
        raise ValueError("expected %d group codes" % GROUPS_PER_BLOCK)
    if int(code_indices.max(initial=0)) >= CODEBOOK_SIZE:
        raise ValueError("code index exceeds 5-bit codebook")
    bits = 0
    for position, code in enumerate(code_indices):
        bits |= int(code) << (CODE_BITS * position)
    return bits.to_bytes(GROUPS_PER_BLOCK * CODE_BITS // 8, "little")


def unpack_codes(payload: bytes) -> np.ndarray:
    bits = int.from_bytes(payload, "little")
    mask = (1 << CODE_BITS) - 1
    return np.array(
        [(bits >> (CODE_BITS * position)) & mask for position in range(GROUPS_PER_BLOCK)],
        dtype=np.int64,
    )


def encode_block(values, importance, patterns):
    groups = values.reshape(-1, LANES)
    weights = importance.reshape(-1, LANES)
    scale = float(np.sum(weights * np.abs(groups)) / max(np.sum(weights), 1e-12))
    for _ in range(3):
        residual = groups[:, None, :] - scale * patterns[None, :, :]
        cost = np.sum(weights[:, None, :] * residual ** 2, axis=2)
        choice = np.argmin(cost, axis=1)
        selected = patterns[choice]
        numerator = float(np.sum(weights * groups * selected))
        denominator = float(np.sum(weights * selected ** 2))
        if denominator > 0.0:
            scale = numerator / denominator
    residual = groups[:, None, :] - scale * patterns[None, :, :]
    cost = np.sum(weights[:, None, :] * residual ** 2, axis=2)
    choice = np.argmin(cost, axis=1)
    half = np.float16(scale)
    return choice.astype(np.int64), float(half), struct.pack("<e", half) + pack_codes(choice)


def decode_block(payload: bytes, patterns) -> np.ndarray:
    if len(payload) != BLOCK_BYTES:
        raise ValueError("block payload must be %d bytes" % BLOCK_BYTES)
    scale = float(struct.unpack("<e", payload[:2])[0])
    codes = unpack_codes(payload[2:])
    return (patterns[codes] * scale).reshape(-1)


def encode_tensor(values, importance, patterns):
    usable = (values.size // QK_BLOCK) * QK_BLOCK
    if usable == 0:
        raise ValueError("tensor needs at least one 256-weight block")
    canonical = canonical_codebook(patterns)
    header = serialize_codebook(canonical)
    blocks = []
    for start in range(0, usable, QK_BLOCK):
        _, _, payload = encode_block(
            values[start : start + QK_BLOCK], importance[start : start + QK_BLOCK], canonical
        )
        blocks.append(payload)
    return header, b"".join(blocks), usable // QK_BLOCK


def decode_tensor(header: bytes, payload: bytes, block_count: int) -> np.ndarray:
    patterns = deserialize_codebook(header)
    if len(payload) != block_count * BLOCK_BYTES:
        raise ValueError("payload size does not match block count")
    out = np.empty(block_count * QK_BLOCK, dtype=np.float64)
    for index in range(block_count):
        chunk = payload[index * BLOCK_BYTES : (index + 1) * BLOCK_BYTES]
        out[index * QK_BLOCK : (index + 1) * QK_BLOCK] = decode_block(chunk, patterns)
    return out


def stored_bytes(numel: int) -> int:
    return (numel // QK_BLOCK) * BLOCK_BYTES + CODEBOOK_BYTES


def bits_per_weight(numel: int) -> float:
    return stored_bytes(numel) * 8.0 / numel


def stq_codebook() -> np.ndarray:
    return ALL_PATTERNS[np.sum(ALL_PATTERNS == 0.0, axis=1) == 1]
