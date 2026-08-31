from __future__ import annotations

import sys
import types

import torch

stub_datasets = types.ModuleType("datasets")
stub_datasets.load_dataset = lambda *a, **k: iter([])
sys.modules.setdefault("datasets", stub_datasets)

stub_tf = types.ModuleType("transformers")
stub_tf.AutoModelForCausalLM = object
stub_tf.AutoTokenizer = object
sys.modules.setdefault("transformers", stub_tf)

stub_iq2 = types.ModuleType("torch_iq2")
stub_iq2.TIERS = {
    "iq2_xxs": {"bpw": 2.0625},
    "iq3_xxs": {"bpw": 3.0625},
    "iq3_s": {"bpw": 3.4375},
}
stub_iq2.quantize_rows = lambda flat, *a, **k: (flat.clone(), 0.0)
sys.modules.setdefault("torch_iq2", stub_iq2)

stub_ltc = types.ModuleType("torch_ltc")
stub_ltc.quantize_rows = lambda flat, *a, **k: (flat.clone(), 0.0)
sys.modules.setdefault("torch_ltc", stub_ltc)

apply_plan = __import__("eval_mixed").apply_plan
build_plans = __import__("eval_tasks").build_plans


class DenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(256, 512, bias=False)
        self.up_proj = torch.nn.Linear(256, 512, bias=False)
        self.down_proj = torch.nn.Linear(512, 256, bias=False)


class DenseLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DenseMLP()


class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([DenseLayer(), DenseLayer()])
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ModuleList([DenseLayer()])
        self.model.visual = torch.nn.Module()
        self.model.visual.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.model.visual.blocks[0].mlp = torch.nn.Module()
        self.model.visual.blocks[0].mlp.linear_fc1 = torch.nn.Linear(256, 512, bias=False)


class ExpertBank(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.ones(2, 4, 256))
        self.down_proj = torch.nn.Parameter(torch.ones(2, 4, 256))


class MoEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].mlp = torch.nn.Module()
        self.model.layers[0].mlp.experts = ExpertBank()


failures = []
plans = build_plans(low_layers=1)
dense_model = DenseModel()
dense_importance = {
    name: torch.ones(module.weight.shape[-1])
    for name, module in dense_model.named_modules()
    if isinstance(module, torch.nn.Linear)
}
expected_params = sum(
    module.weight.numel()
    for name, module in dense_model.named_modules()
    if name.startswith("model.language_model.layers.") and isinstance(module, torch.nn.Linear)
)

dense_iq3_stats = apply_plan(dense_model, dense_importance, plans["dense_iq3"], "cpu")
if dense_iq3_stats["params"] != expected_params:
    failures.append("dense_iq3 params %d != %d" % (dense_iq3_stats["params"], expected_params))
if dense_iq3_stats["bpw"] != 3.0625:
    failures.append("dense_iq3 bpw %.4f != 3.0625" % dense_iq3_stats["bpw"])
inventory = dense_iq3_stats.get("tensor_inventory")
if not isinstance(inventory, list) or len(inventory) != 6:
    failures.append("dense_iq3 did not record the six processed tensor identities")
else:
    required_inventory_fields = {
        "module_name", "weight_name", "layer", "attribute", "tier", "params", "bpw"
    }
    if any(not required_inventory_fields.issubset(record) for record in inventory):
        failures.append("dense_iq3 tensor inventory is missing identity or tier fields")
    if [record["weight_name"] for record in inventory] != sorted(
            record["weight_name"] for record in inventory):
        failures.append("dense_iq3 tensor inventory is not deterministic")
    if {record["layer"] for record in inventory} != {0, 1}:
        failures.append("dense_iq3 tensor inventory did not bind layers")
    if {record["attribute"] for record in inventory} != {
            "gate_proj", "up_proj", "down_proj"}:
        failures.append("dense_iq3 tensor inventory did not bind projection attributes")
    if any(record["tier"] != "iq3_xxs" or record["bpw"] != 3.0625
           for record in inventory):
        failures.append("dense_iq3 tensor inventory did not bind tier and bpw")
if dense_iq3_stats.get("skipped_targets") != []:
    failures.append("complete dense importance unexpectedly recorded skipped targets")

missing_importance = dict(dense_importance)
missing_name = "model.language_model.layers.1.mlp.down_proj"
del missing_importance[missing_name]
missing_stats = apply_plan(DenseModel(), missing_importance, plans["dense_iq3"], "cpu")
if missing_stats.get("params") == expected_params:
    failures.append("missing importance silently retained the complete parameter count")
if missing_stats.get("skipped_targets") != [{
    "module_name": missing_name,
    "weight_name": missing_name + ".weight",
    "layer": 1,
    "attribute": "down_proj",
    "reason": "missing_importance",
}]:
    failures.append("missing importance did not produce the canonical skipped-target record")

dense_iq3s_stats = apply_plan(DenseModel(), dense_importance, plans["dense_iq3s"], "cpu")
if dense_iq3s_stats["bpw"] != 3.4375:
    failures.append("dense_iq3s bpw %.4f != 3.4375" % dense_iq3s_stats["bpw"])


def layer_plan(layer, _attribute):
    return "iq3_s" if layer == 1 else "fp16"


layer_stats = apply_plan(DenseModel(), dense_importance, layer_plan, "cpu")
if not 3.4375 < layer_stats["bpw"] < 16:
    failures.append("layer plan bpw %.4f did not exercise both layers" % layer_stats["bpw"])

moe_model = MoEModel()
moe_importance = {"model.layers.0.mlp.experts": torch.ones(256)}
moe_stats = apply_plan(moe_model, moe_importance, plans["mixed_ltc"], "cpu")
if moe_stats["params"] == 0:
    failures.append("mixed_ltc did not quantize the MoE expert parameters")

for arm, tier in (
        ("dense_iq2", "iq2_xxs"),
        ("dense_iq3", "iq3_xxs"),
        ("dense_iq3s", "iq3_s"),
        ("dense_fp16", "fp16")):
    for attribute in ("gate_proj", "up_proj", "down_proj"):
        actual = plans[arm](0, attribute)
        if actual != tier:
            failures.append("%s %s returned %s" % (arm, attribute, actual))

if failures:
    for line in failures:
        print("FAIL: " + line)
    raise SystemExit(1)
print("PASS: dense MLP plans quantize six scoped tensors and preserve the MoE path")
