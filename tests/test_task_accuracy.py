from __future__ import annotations

import random

from task_accuracy import compare, mcnemar_exact, paired_accuracy_bootstrap

failures = []

if abs(mcnemar_exact(0, 0) - 1.0) > 1e-12:
    failures.append("no disagreement must give p=1.0")

symmetric = mcnemar_exact(10, 10)
if symmetric < 0.9:
    failures.append("symmetric disagreement should be far from significant, got %.4f" % symmetric)

lopsided = mcnemar_exact(12, 0)
if lopsided > 0.001:
    failures.append("12 vs 0 should be highly significant, got %.6f" % lopsided)

if not (mcnemar_exact(9, 1) < 0.05):
    failures.append("9 vs 1 should be significant, got %.4f" % mcnemar_exact(9, 1))

if abs(mcnemar_exact(3, 7) - mcnemar_exact(7, 3)) > 1e-12:
    failures.append("mcnemar must be symmetric in its arguments")

identical = [1, 0, 1, 1, 0] * 20
boot_same = paired_accuracy_bootstrap(identical, identical)
if abs(boot_same["delta"]) > 1e-12:
    failures.append("identical arms must have zero delta")
if boot_same["excludes_zero"]:
    failures.append("identical arms must not exclude zero")

rng = random.Random(5)
strong = [1] * 80 + [0] * 20
weak = [1 if rng.random() < 0.55 else 0 for _ in range(100)]
boot_diff = paired_accuracy_bootstrap(strong, weak)
if not boot_diff["excludes_zero"]:
    failures.append("80%% vs ~55%% should exclude zero, CI [%.3f, %.3f]" % (
        boot_diff["ci_low"], boot_diff["ci_high"]))
if boot_diff["delta"] <= 0:
    failures.append("stronger arm must show positive delta")

report = compare({"dense": [1] * 50, "arm_a": [1] * 40 + [0] * 10,
                  "arm_b": [1] * 30 + [0] * 20}, "dense")
if report["items"] != 50:
    failures.append("item count wrong")
if abs(report["arms"]["arm_a"]["accuracy"] - 0.8) > 1e-12:
    failures.append("arm_a accuracy should be 0.80")
key = "arm_a_vs_arm_b"
if key not in report["comparisons"]:
    failures.append("missing pairwise comparison")
else:
    cmp = report["comparisons"][key]
    if cmp["only_first_correct"] != 10 or cmp["only_second_correct"] != 0:
        failures.append("discordant counts wrong: %s" % cmp)
baseline_key = "dense_vs_arm_a"
if baseline_key not in report["comparisons"]:
    failures.append("missing baseline comparison")
elif report["comparisons"][baseline_key]["only_first_correct"] != 10:
    failures.append("baseline discordant count wrong")
if len(report["comparisons"]) != 3:
    failures.append("three arms must produce three pairwise comparisons")
reordered = compare({"arm_a": [0, 1], "dense": [1, 1]}, "dense")
if "dense_vs_arm_a" not in reordered["comparisons"]:
    failures.append("baseline must be first even when input order differs")

try:
    compare({"dense": [1, 0], "bad": [1, 0, 1]}, "dense")
    failures.append("mismatched arm lengths must raise")
except ValueError:
    pass

try:
    compare({"arm": [1, 0]}, "dense")
    failures.append("missing baseline must raise")
except ValueError:
    pass

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)

print("PASS: task-accuracy statistics verified")
print("  mcnemar: p=1.0 at no disagreement, symmetric, 12v0 highly significant")
print("  bootstrap: identical arms give zero delta and include zero")
print("  bootstrap: 80%% vs 55%% excludes zero with positive delta")
print("  all three pairs include baseline, discordant counting exact (10 vs 0)")
print("  mismatched lengths and missing baseline both raise")
