from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")


def audit(imatrix_path: Path, minimum_hits: int) -> dict:
    payload = json.loads(imatrix_path.read_text(encoding="utf-8"))
    routing = payload.get("routing", {})
    if not routing:
        raise RuntimeError("imatrix has no routing section; re-collect with router hooks")

    starved = []
    per_layer = {}
    all_min = []
    for name, stats in sorted(routing.items()):
        match = LAYER_PATTERN.search(name)
        layer = int(match.group(1)) if match else -1
        hits = stats.get("hits") or []
        per_layer[layer] = len(hits)
        all_min.append(min(hits) if hits else 0)
        for expert, count in enumerate(hits):
            if count < minimum_hits:
                starved.append({"layer": layer, "expert": expert, "topk_hits": count})

    report = {
        "model_id": payload.get("model_id"),
        "revision": payload.get("revision"),
        "observed_tokens": payload.get("observed_tokens"),
        "domains": payload.get("domains"),
        "layers_seen": len(routing),
        "experts_per_layer": per_layer,
        "expert_pairs": sum(per_layer.values()),
        "minimum_hits_required": minimum_hits,
        "starved_experts": starved[:32],
        "starved_count": len(starved),
        "min_hits_observed": min(all_min) if all_min else 0,
        "coverage_ok": len(starved) == 0 and len(routing) > 0,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="audit per-expert routing coverage")
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--minimum-hits", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = audit(Path(args.imatrix), args.minimum_hits)
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if not report["coverage_ok"]:
        print("COVERAGE FAILED: %d starved experts" % report["starved_count"])
        return 1
    print("COVERAGE OK: %d layer x expert pairs, min hits %d" % (
        report["expert_pairs"], report["min_hits_observed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

