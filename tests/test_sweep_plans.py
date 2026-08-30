from __future__ import annotations

import sys
import types

stub_datasets = types.ModuleType("datasets")
stub_datasets.load_dataset = lambda *a, **k: iter([])
sys.modules.setdefault("datasets", stub_datasets)

stub_tf = types.ModuleType("transformers")
stub_tf.AutoModelForCausalLM = object
stub_tf.AutoTokenizer = object
sys.modules.setdefault("transformers", stub_tf)

stub_iq2 = types.ModuleType("torch_iq2")
stub_iq2.quantize_rows = lambda *a, **k: (None, 0.0)
sys.modules.setdefault("torch_iq2", stub_iq2)
stub_ltc = types.ModuleType("torch_ltc")
stub_ltc.quantize_rows = lambda *a, **k: (None, 0.0)
sys.modules.setdefault("torch_ltc", stub_ltc)

build_plans = __import__("eval_tasks").build_plans

failures = []

plans = build_plans(low_layers=6)
for name in ("dense", "uniform_iq2", "mixed_ltc", "iq2s_all", "iq3_all", "ltc_iq3"):
    if name not in plans:
        failures.append("plan %s missing" % name)

if "ltc_iq3" in plans:
    plan = plans["ltc_iq3"]
    if plan(0, "gate_up_proj") != "ltc":
        failures.append("ltc_iq3 must use ltc on low gate_up, got %s" % plan(0, "gate_up_proj"))
    if plan(9, "gate_up_proj") != "iq3_xxs":
        failures.append("ltc_iq3 must use iq3_xxs above the band, got %s" % plan(9, "gate_up_proj"))
    if plan(0, "down_proj") != "iq3_xxs":
        failures.append("ltc_iq3 must use iq3_xxs on down_proj, got %s" % plan(0, "down_proj"))

if "iq3_all" in plans and plans["iq3_all"](0, "gate_up_proj") != "iq3_xxs":
    failures.append("iq3_all must be uniform iq3_xxs")

if "dense" in plans and plans["dense"](0, "gate_up_proj") != "fp16":
    failures.append("dense must report fp16")

if failures:
    for line in failures:
        print("FAIL: " + line)
    raise SystemExit(1)
print("PASS: six sweep plans defined with the expected tier per layer band")
