from __future__ import annotations

import torch

TARGET_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


class Block(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, hidden, bias=False)
        self.up_proj = torch.nn.Linear(dim, hidden, bias=False)
        self.down_proj = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


model = torch.nn.Sequential(Block(16, 32), Block(16, 32))
sums, counts = {}, {}


def hook(name):
    def inner(_module, inputs, _output):
        activation = inputs[0].detach()
        flat = activation.reshape(-1, activation.shape[-1]).float()
        squared = (flat * flat).sum(dim=0)
        if name in sums:
            sums[name] += squared
            counts[name] += flat.shape[0]
        else:
            sums[name] = squared
            counts[name] = flat.shape[0]
    return inner


handles = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
        handles.append(module.register_forward_hook(hook(name)))

print("hooked layers:", len(handles))
torch.manual_seed(0)
batches = [torch.randn(2, 7, 16) for _ in range(3)]
with torch.no_grad():
    for b in batches:
        model(b)

for h in handles:
    h.remove()

failures = []
if len(sums) != 6:
    failures.append("expected 6 hooked linears, got %d" % len(sums))

expected_rows = sum(b.shape[0] * b.shape[1] for b in batches)
for name, count in counts.items():
    if count != expected_rows:
        failures.append("%s row count %d != %d" % (name, count, expected_rows))

stacked = torch.cat([b.reshape(-1, 16) for b in batches], dim=0)
manual = (stacked * stacked).mean(dim=0)
first = sums["0.gate_proj"] / counts["0.gate_proj"]
if not torch.allclose(first, manual, atol=1e-5):
    failures.append("gate_proj E[x^2] mismatch: max delta %.3e" % float((first - manual).abs().max()))

if sums["0.down_proj"].numel() != 32:
    failures.append("down_proj should see hidden dim 32, got %d" % sums["0.down_proj"].numel())

for name, total in sums.items():
    mean_square = total / counts[name]
    if not bool(torch.all(mean_square >= 0)):
        failures.append("%s produced negative mean square" % name)
    if not bool(torch.all(torch.isfinite(mean_square))):
        failures.append("%s produced non-finite values" % name)

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)

print("PASS: imatrix hook math verified on CPU")
print("  6 linears hooked (gate/up/down x 2 blocks)")
print("  row accounting exact: %d rows" % expected_rows)
print("  E[x^2] matches manual computation (atol 1e-5)")
print("  down_proj correctly sees hidden dim (32), gate/up see model dim (16)")
print("  all values finite and non-negative")

