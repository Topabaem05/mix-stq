from __future__ import annotations

import json
import re
from pathlib import Path

source = Path("ggml-common.h").read_text(encoding="utf-8")


def table(name: str) -> list[int]:
    match = re.search(
        r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + name + r"\s*,\s*(\d+)\s*\)(.*?)GGML_TABLE_END\(\)",
        source,
        re.S,
    )
    if match is None:
        raise RuntimeError("table not found: " + name)
    declared = int(match.group(1))
    body = match.group(2)
    hex_values = re.findall(r"0x([0-9a-fA-F]+)", body)
    if hex_values:
        values = [int(v, 16) for v in hex_values]
    else:
        values = [int(v) for v in re.findall(r"(?<![\w.])(\d+)(?![\w.])", body)]
    if len(values) != declared:
        raise RuntimeError("%s: declared %d got %d" % (name, declared, len(values)))
    return values


grid = table("iq2xxs_grid")
ksigns = table("ksigns_iq2xs")
kmask = table("kmask_iq2xs")

points = []
for packed in grid:
    entry = [(packed >> (8 * j)) & 0xFF for j in range(8)]
    points.append(entry)

flat = [v for e in points for v in e]
print("iq2xxs_grid entries:", len(points))
print("values per entry:", len(points[0]))
print("distinct magnitudes in grid:", sorted(set(flat)))
print("ksigns len:", len(ksigns), "kmask:", kmask)
print("first grid entry:", points[0])
print("last grid entry:", points[-1])

Path("iq2xxs_tables.json").write_text(
    json.dumps({"grid": points, "ksigns": ksigns, "kmask": kmask}), encoding="utf-8"
)
print("wrote iq2xxs_tables.json")

QK_K = 256
block_bytes = 2 + (QK_K // 8) * 2
print("block_iq2_xxs bytes: %d (d=2, qs=%d x uint16)" % (block_bytes, QK_K // 8))
print("computed bpw from block layout: %.4f" % (block_bytes * 8.0 / QK_K))
