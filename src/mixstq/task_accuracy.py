from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    smaller = min(only_a, only_b)
    total = 0.0
    for k in range(smaller + 1):
        total += math.comb(n, k)
    probability = 2.0 * total / (2.0 ** n)
    return min(1.0, probability)


def paired_accuracy_bootstrap(correct_a, correct_b, iterations=10000, seed=22):
    if len(correct_a) != len(correct_b):
        raise ValueError("arms must cover the same items")
    rng = random.Random(seed)
    count = len(correct_a)
    if count == 0:
        raise ValueError("no items")
    deltas = []
    indices = range(count)
    for _ in range(iterations):
        picks = [rng.randrange(count) for _ in indices]
        a = sum(correct_a[i] for i in picks) / count
        b = sum(correct_b[i] for i in picks) / count
        deltas.append(a - b)
    deltas.sort()
    low = deltas[int(0.025 * iterations)]
    high = deltas[int(0.975 * iterations) - 1]
    observed = sum(correct_a) / count - sum(correct_b) / count
    return {
        "delta": observed,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def compare(records: dict[str, list[int]], baseline: str) -> dict:
    if baseline not in records:
        raise ValueError("baseline arm %s missing" % baseline)
    base = records[baseline]
    report = {"baseline": baseline, "items": len(base), "arms": {}, "comparisons": {}}
    for name, correct in records.items():
        if len(correct) != len(base):
            raise ValueError("arm %s has %d items, baseline has %d" % (name, len(correct), len(base)))
        report["arms"][name] = {
            "correct": sum(correct),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
        }
    names = [n for n in records if n != baseline]
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a = records[first]
            b = records[second]
            only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
            only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
            boot = paired_accuracy_bootstrap(a, b)
            report["comparisons"]["%s_vs_%s" % (first, second)] = {
                "only_first_correct": only_a,
                "only_second_correct": only_b,
                "mcnemar_p": mcnemar_exact(only_a, only_b),
                "accuracy_delta": boot["delta"],
                "ci_95": [boot["ci_low"], boot["ci_high"]],
                "significant": boot["excludes_zero"] and mcnemar_exact(only_a, only_b) < 0.05,
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="paired task-accuracy comparison with McNemar and bootstrap"
    )
    parser.add_argument("--results", required=True,
                        help="json mapping arm name -> list of 0/1 per item")
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    records = {name: [int(bool(v)) for v in values] for name, values in payload.items()}
    report = compare(records, args.baseline)

    print("items: %d" % report["items"])
    for name, stats in report["arms"].items():
        print("  %-18s %4d/%d  %.4f" % (name, stats["correct"], report["items"], stats["accuracy"]))
    print()
    for label, cmp in report["comparisons"].items():
        print("%-34s delta=%+.4f CI[%+.4f, %+.4f] mcnemar_p=%.4f %s" % (
            label, cmp["accuracy_delta"], cmp["ci_95"][0], cmp["ci_95"][1],
            cmp["mcnemar_p"], "SIGNIFICANT" if cmp["significant"] else "not significant"))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print()
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

