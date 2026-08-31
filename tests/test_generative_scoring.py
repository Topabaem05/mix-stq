from __future__ import annotations

import importlib
import sys
import types

import torch


def empty_dataset(*args, **kwargs):
    return iter([])


def empty_quantization(*args, **kwargs):
    return None, 0.0


stub_datasets = types.ModuleType("datasets")
stub_datasets.load_dataset = empty_dataset
sys.modules.setdefault("datasets", stub_datasets)

stub_tf = types.ModuleType("transformers")
stub_tf.AutoModelForCausalLM = object
stub_tf.AutoTokenizer = object
sys.modules.setdefault("transformers", stub_tf)

stub_iq2 = types.ModuleType("torch_iq2")
stub_iq2.quantize_rows = empty_quantization
sys.modules.setdefault("torch_iq2", stub_iq2)
stub_ltc = types.ModuleType("torch_ltc")
stub_ltc.quantize_rows = empty_quantization
sys.modules.setdefault("torch_ltc", stub_ltc)

generative = importlib.import_module("eval_generative")
extract_prediction = generative.extract_prediction
parse_gold_letter = generative.parse_gold_letter
slice_continuations = generative.slice_continuations

failures = []

for letter in "ABCD":
    parsed = parse_gold_letter("\\boxed{%s}" % letter)
    if parsed != letter:
        failures.append("gold parser returned %r for %s" % (parsed, letter))

multiple = "The format can look like \\boxed{A}. After solving, the result is \\boxed{C}"
if extract_prediction(multiple) != "C":
    failures.append("prediction parser did not choose the last boxed answer")

thinking = "<think>\nmaybe \\boxed{B}\n</think>\n\nFinal: \\boxed{D}"
if extract_prediction(thinking) != "D":
    failures.append("prediction parser did not survive a thinking block")

if extract_prediction("After checking the work, the answer is B.") != "B":
    failures.append("answer-is fallback did not parse B")

if extract_prediction("No option can be determined from this response.") is not None:
    failures.append("unparsable text did not return None")

if extract_prediction("Final: \\boxed{c}") != "C":
    failures.append("lowercase boxed answer was not normalized")

generated = torch.tensor([
    [0, 0, 11, 12, 91, 92],
    [21, 22, 23, 24, 93, 94],
])
continuations = slice_continuations(generated, [2, 4], 4)
if not torch.equal(continuations[0], torch.tensor([91, 92])):
    failures.append("short prompt continuation included left padding or prompt tokens")
if not torch.equal(continuations[1], torch.tensor([93, 94])):
    failures.append("long prompt continuation was sliced at another row's boundary")

if failures:
    print("FAIL")
    for failure in failures:
        print("  " + failure)
    raise SystemExit(1)

print("PASS: generative scoring and batched continuation slicing verified offline")
