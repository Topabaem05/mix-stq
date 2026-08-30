from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def paired_bootstrap(left, right, iterations=10000, seed=22):
    rng = np.random.default_rng(seed)
    left_nll = np.array([d["nll_sum"] for d in left], dtype=np.float64)
    right_nll = np.array([d["nll_sum"] for d in right], dtype=np.float64)
    tokens = np.array([d["tokens"] for d in left], dtype=np.float64)
    count = len(tokens)
    deltas = np.empty(iterations, dtype=np.float64)
    for step in range(iterations):
        pick = rng.integers(0, count, count)
        a = left_nll[pick].sum() / tokens[pick].sum()
        b = right_nll[pick].sum() / tokens[pick].sum()
        deltas[step] = math.exp(b) - math.exp(a)
    observed = math.exp(right_nll.sum() / tokens.sum()) - math.exp(left_nll.sum() / tokens.sum())
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "observed_ppl_delta": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "documents": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="paired bootstrap over per-document NLL")
    parser.add_argument("--results", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    left = payload[args.left].get("per_document")
    right = payload[args.right].get("per_document")
    if not left or not right:
        raise RuntimeError("per_document records missing; re-run the evaluation")
    report = paired_bootstrap(left, right)
    report["left"] = args.left
    report["right"] = args.right
    print(json.dumps(report, indent=1))
    verdict = "significant" if report["excludes_zero"] else "not significant"
    print("%s vs %s: ppl delta %+.3f, 95%% CI [%+.3f, %+.3f] -> %s" % (
        args.left, args.right, report["observed_ppl_delta"],
        report["ci_lower"], report["ci_upper"], verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

