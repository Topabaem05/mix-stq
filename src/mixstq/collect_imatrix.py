from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def collect(model_id: str, revision: str, samples: int, seq_len: int, out: Path,
            domains: tuple[str, ...]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def hook(name: str):
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
    if not handles:
        raise RuntimeError("no target linear layers matched")

    texts: list[str] = []
    per_domain = max(samples // max(len(domains), 1), 1)
    for domain in domains:
        if domain == "wiki":
            stream = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
            field = "text"
        elif domain == "code":
            stream = load_dataset("codeparrot/github-code", split="train", streaming=True,
                                  languages=["Python"], licenses=["mit"], trust_remote_code=True)
            field = "code"
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

    observed_tokens = 0
    with torch.no_grad():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            model(**batch)
            observed_tokens += int(batch["input_ids"].numel())

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
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    store = {}
    for name, total in sums.items():
        mean_square = (total / max(counts[name], 1)).cpu()
        store[name] = mean_square
        payload["tensors"][name] = {
            "channels": int(mean_square.numel()),
            "rows_observed": counts[name],
            "mean": float(mean_square.mean()),
            "max": float(mean_square.max()),
            "min": float(mean_square.min()),
        }
    torch.save(store, out.with_suffix(".pt"))
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="collect a real activation importance matrix")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--domains", default="wiki,code,chat")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = collect(
        args.model, args.revision, args.samples, args.seq_len, Path(args.out),
        tuple(d.strip() for d in args.domains.split(",") if d.strip()),
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "tensors"}, indent=1))
    print("tensors captured:", len(payload["tensors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

