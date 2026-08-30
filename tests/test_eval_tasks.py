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
stub_iq2.quantize_rows = lambda *a, **k: (None, 0.0)
sys.modules.setdefault("torch_iq2", stub_iq2)
stub_ltc = types.ModuleType("torch_ltc")
stub_ltc.quantize_rows = lambda *a, **k: (None, 0.0)
sys.modules.setdefault("torch_ltc", stub_ltc)

from eval_tasks import LETTERS, render, score_item  # noqa: E402

failures = []

item = {
    "task": "mmlu",
    "subject": "test",
    "question": "What is 2+2?",
    "choices": ["3", "4", "5", "6"],
    "answer": 1,
}
text = render(item)
for letter in LETTERS:
    if ("%s. " % letter) not in text:
        failures.append("rendered prompt missing option %s" % letter)
if not text.rstrip().endswith("Answer:"):
    failures.append("prompt must end with the answer cue")
if item["question"] not in text:
    failures.append("prompt missing the question")


class StubTokenizer:
    def __init__(self, favored):
        self.favored = favored

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = [1, 2, 3] if return_tensors else [10 + LETTERS.index(text.strip())]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


class StubModel:
    def __init__(self, favored_token):
        self.favored_token = favored_token

    def __call__(self, input_ids=None):
        length = input_ids.shape[1]
        logits = torch.full((1, length, 32), -5.0)
        logits[0, :, self.favored_token] = 5.0
        return types.SimpleNamespace(logits=logits)


for expected_index, letter in enumerate(LETTERS):
    tokenizer = StubTokenizer(letter)
    model = StubModel(10 + expected_index)
    picked = score_item(model, tokenizer, item, "cpu")
    if picked != expected_index:
        failures.append("scoring picked %d when option %s was favored" % (picked, letter))

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)

print("PASS: task scoring verified offline")
print("  prompt renders all four options and the answer cue")
print("  argmax over per-letter log-likelihood picks the favored option in all 4 cases")
print("  no GPU, no network, no model download required")
