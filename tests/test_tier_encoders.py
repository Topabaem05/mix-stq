from __future__ import annotations

import torch
import torch_iq2 as tq

failures = []

expected_bpw = {"iq2_xxs": 2.0625, "iq2_s": 2.5625, "iq3_xxs": 3.0625}
for tier, bpw in expected_bpw.items():
    if tier not in tq.TIERS:
        failures.append("tier %s not registered" % tier)
        continue
    if abs(tq.TIERS[tier]["bpw"] - bpw) > 1e-9:
        failures.append("tier %s bpw %.4f expected %.4f" % (tier, tq.TIERS[tier]["bpw"], bpw))

expected_lane = {"iq2_xxs": 8, "iq2_s": 8, "iq3_xxs": 4}
for tier, lane in expected_lane.items():
    if tier in tq.TIERS and tq.TIERS[tier]["lane"] != lane:
        failures.append("tier %s lane %d expected %d" % (tier, tq.TIERS[tier]["lane"], lane))

torch.manual_seed(0)
matrix = torch.randn(8, 256)
importance = torch.ones(256)
errors = {}
for tier in expected_bpw:
    if tier not in tq.TIERS:
        continue
    quantized, relative = tq.quantize_rows(matrix, importance, tier=tier)
    errors[tier] = relative
    if quantized.shape != matrix.shape:
        failures.append("tier %s changed shape" % tier)
    if not (0.0 < relative < 1.0):
        failures.append("tier %s relative error %.6f out of range" % (tier, relative))

if len(errors) == 3 and not (errors["iq3_xxs"] < errors["iq2_s"] < errors["iq2_xxs"]):
    failures.append("error must decrease as bits increase: %s" % errors)

if failures:
    for line in failures:
        print("FAIL: " + line)
    raise SystemExit(1)
print("PASS: three tiers registered, bpw exact, error monotone in bits")
