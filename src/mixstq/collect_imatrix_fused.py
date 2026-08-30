from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERT_MODULE_NAMES = ("mlp.experts",)
ROUTER_MODULE_NAMES = ("mlp.gate",)


def _pick_texts(domains, per_domain):
    texts = []
    for domain in domains:
        if domain == "wiki":
            stream = load_dataset(
                "Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True
            )
            field = "text"
        elif domain == "code":
            stream = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
            field = "content"
        elif domain == "chat":
            stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
            field = "prompt"
        else:
            raise RuntimeError("unknown domain " + domain)
        taken = 0
        for row in stream:
            text = (row.get(field) or "").strip()
            if len(text) < 200:
                continue
            texts.append(text)
            taken += 1
            if taken >= per_domain:
                break
    return texts


def collect(model_id, revision, samples, seq_len, out, domains):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    input_sums = {}
    input_counts = {}
    router_mass = {}
    router_counts = {}

    def expert_hook(name):
        def inner(_module, inputs, _output):
            activation = inputs[0].detach()
            flat = activation.reshape(-1, activation.shape[-1]).float()
            squared = (flat * flat).sum(dim=0)
            if name in input_sums:
                input_sums[name] += squared
                input_counts[name] += flat.shape[0]
            else:
                input_sums[name] = squared
                input_counts[name] = flat.shape[0]
        return inner

    def router_hook(name):
        def inner(_module, _inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(8, probs.shape[-1]), dim=-1)
            counts = torch.zeros(probs.shape[-1], device=probs.device)
            counts.scatter_add_(0, top.indices.reshape(-1), torch.ones_like(top.values.reshape(-1)))
            if name in router_mass:
                router_mass[name] += probs.sum(dim=0).cpu()
                router_counts[name] += counts.cpu()
            else:
                router_mass[name] = probs.sum(dim=0).cpu()
                router_counts[name] = counts.cpu()
        return inner

    handles = []
    for name, module in model.named_modules():
        if name.endswith(EXPERT_MODULE_NAMES):
            handles.append(module.register_forward_hook(expert_hook(name)))
        elif name.endswith(ROUTER_MODULE_NAMES):
            handles.append(module.register_forward_hook(router_hook(name)))
    if not handles:
        raise RuntimeError("no expert or router modules matched")
    print("hooked %d modules" % len(handles), flush=True)

    texts = _pick_texts(domains, max(samples // max(len(domains), 1), 1))
    print("collected %d documents" % len(texts), flush=True)

    observed_tokens = 0
    with torch.no_grad():
        for index, text in enumerate(texts):
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            model(**batch)
            observed_tokens += int(batch["input_ids"].numel())
            if (index + 1) % 24 == 0:
                print("  %d/%d docs, %d tokens" % (index + 1, len(texts), observed_tokens), flush=True)

    for handle in handles:
        handle.remove()

    payload = {
        "model_id": model_id,
        "revision": revision,
        "domains": list(domains),
        "documents": len(texts),
        "observed_tokens": observed_tokens,
        "sequence_length": seq_len,
        "tensors": {},
        "routing": {},
    }
    store = {}
    for name, total in input_sums.items():
        mean_square = (total / max(input_counts[name], 1)).cpu()
        store[name] = mean_square
        payload["tensors"][name] = {
            "channels": int(mean_square.numel()),
            "rows_observed": input_counts[name],
            "mean": float(mean_square.mean()),
            "max": float(mean_square.max()),
            "min": float(mean_square.min()),
        }
    for name, mass in router_mass.items():
        counts = router_counts[name]
        total_hits = float(counts.sum())
        payload["routing"][name] = {
            "experts": int(counts.numel()),
            "total_topk_hits": total_hits,
            "min_expert_hits": float(counts.min()),
            "max_expert_hits": float(counts.max()),
            "starved_experts": int((counts < 32).sum()),
            "probability_mass": [round(float(v), 6) for v in (mass / max(mass.sum(), 1e-9))],
            "hits": [int(v) for v in counts],
        }
        store[name + ".router_hits"] = counts

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, out.with_suffix(".pt"))
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="collect activation importance and routing stats")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--domains", default="wiki,code,chat")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = collect(
        args.model, args.revision, args.samples, args.seq_len, Path(args.out),
        tuple(d.strip() for d in args.domains.split(",") if d.strip()),
    )
    summary = {k: v for k, v in payload.items() if k not in ("tensors", "routing")}
    print(json.dumps(summary, indent=1))
    print("expert-input tensors:", len(payload["tensors"]))
    print("routers:", len(payload["routing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
