from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_iq2 as tq
import torch_ltc as tl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

GATE_UP = "gate_up_proj"
DOWN = "down_proj"


def held_out(count, minimum_chars=400):
    stream = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    texts = []
    for row in stream:
        text = (row.get("text") or "").strip()
        if len(text) >= minimum_chars:
            texts.append(text)
        if len(texts) >= count:
            break
    return texts


def uniform_iq2_plan(_layer, _attribute):
    return "iq2"


def reference_pass(model, tokenizer, texts, seq_len, device):
    records = []
    with torch.no_grad():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            ids = batch["input_ids"].to(device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            targets = ids[0, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            top2 = logits.topk(2, dim=-1).values
            records.append({
                "ids": ids.cpu(),
                "log_probs": log_probs.cpu(),
                "top1": logits.argmax(dim=-1).cpu(),
                "nll_sum": float(nll.sum()),
                "tokens": int(targets.numel()),
                "margin": float((top2[:, 0] - top2[:, 1]).mean()),
            })
    return records


def measure(model, tokenizer, records, device):
    nll_sum, tokens, kl_sum, agree, compared = 0.0, 0, 0.0, 0, 0
    ratios = []
    per_document = []
    with torch.no_grad():
        for record in records:
            ids = record["ids"].to(device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            targets = ids[0, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            nll_sum += float(nll.sum())
            tokens += int(targets.numel())
            kl_sum += float(torch.nn.functional.kl_div(
                log_probs, record["log_probs"].to(device), log_target=True, reduction="batchmean"))
            top1 = logits.argmax(dim=-1)
            agree += int((top1 == record["top1"].to(device)).sum())
            compared += int(top1.numel())
            top2 = logits.topk(2, dim=-1).values
            if record["margin"] > 0:
                ratios.append(float((top2[:, 0] - top2[:, 1]).mean()) / record["margin"])
            per_document.append({
                "nll_sum": float(nll.sum()),
                "tokens": int(targets.numel()),
                "agree": int((top1 == record["top1"].to(device)).sum()),
            })
    mean_nll = nll_sum / max(tokens, 1)
    return {
        "nll_per_token": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "logit_kl": kl_sum / max(len(records), 1),
        "top1_agreement": agree / max(compared, 1),
        "margin_ratio": sum(ratios) / max(len(ratios), 1),
        "tokens": tokens,
        "per_document": per_document,
    }


def router_topk(model, tokenizer, texts, seq_len, device, limit=6):
    captured = {}
    handles = []

    def hook(name):
        def inner(_module, _inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            top = torch.softmax(logits, dim=-1).topk(min(8, logits.shape[-1]), dim=-1).indices
            captured.setdefault(name, []).append(top.cpu())
        return inner

    for name, module in model.named_modules():
        if name.endswith("mlp.gate"):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        for text in texts[:limit]:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            model(input_ids=batch["input_ids"].to(device))
    for handle in handles:
        handle.remove()
    return {k: torch.cat(v, dim=0) for k, v in captured.items()}


def jaccard(reference, candidate):
    scores = []
    for name, ref in reference.items():
        cand = candidate.get(name)
        if cand is None or cand.shape != ref.shape:
            continue
        stride = max(ref.shape[0] // 400, 1)
        for row in range(0, ref.shape[0], stride):
            a = set(int(v) for v in ref[row])
            b = set(int(v) for v in cand[row])
            scores.append(len(a & b) / max(len(a | b), 1))
    return sum(scores) / max(len(scores), 1)


def apply_plan(model, importance, plan, device):
    stats = {"bytes": 0, "params": 0, "errors": []}
    with torch.no_grad():
        for name, module in model.named_modules():
            if not name.endswith("mlp.experts"):
                continue
            layer = int(name.split(".")[2])
            channel = importance.get(name)
            if channel is None:
                continue
            channel = channel.to(device)
            for attribute in (GATE_UP, DOWN):
                param = getattr(module, attribute, None)
                if param is None:
                    continue
                tier = plan(layer, attribute)
                data = param.data
                flat = data.reshape(-1, data.shape[-1])
                width = flat.shape[-1]
                numel = int(data.numel())
                if tier == "fp16":
                    stats["bytes"] += numel * 2
                    stats["params"] += numel
                    continue
                if channel.numel() >= width:
                    local = channel[:width]
                else:
                    local = channel.repeat(width // channel.numel() + 1)[:width]
                if tier == "ltc":
                    quantized, relative = tl.quantize_rows(flat, local, learn=True)
                    bpw = 1.3125
                elif tier == "stq":
                    quantized, relative = tl.quantize_rows(
                        flat, local, patterns=tl.stq_patterns(torch.device(device)), learn=False)
                    bpw = 1.3125
                elif tier == "iq2":
                    quantized, relative = tq.quantize_rows(flat, local, tier="iq2_xxs")
                    bpw = tq.TIERS["iq2_xxs"]["bpw"]
                elif tier in tq.TIERS:
                    quantized, relative = tq.quantize_rows(flat, local, tier=tier)
                    bpw = tq.TIERS[tier]["bpw"]
                else:
                    raise RuntimeError("unknown tier " + tier)
                param.data = quantized.reshape(data.shape).to(data.dtype)
                stats["bytes"] += int(numel * bpw / 8)
                stats["params"] += numel
                stats["errors"].append(relative)
                print("    L%02d %s -> %s err %.4f" % (layer, attribute, tier, relative),
                      flush=True)
    stats["mean_error"] = sum(stats["errors"]) / max(len(stats["errors"]), 1)
    stats["bpw"] = stats["bytes"] * 8.0 / max(stats["params"], 1)
    del stats["errors"]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="mixed precision quality evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--documents", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--low-layers", type=int, default=6)
    parser.add_argument("--arms", default="dense,uniform_iq2,mixed_stq,mixed_ltc")
    parser.add_argument("--sweep-low-layers", default=None,
                        help="comma separated low-layer counts; adds ltc_lowN arms")
    parser.add_argument("--layer-set", action="append", default=None,
                        help="name=1,2,3 explicit LTC layer set; repeatable")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    texts = held_out(args.documents)
    store = torch.load(args.imatrix, map_location="cpu")
    importance = {k: v for k, v in store.items() if not k.endswith(".router_hits")}
    low = set(range(args.low_layers))

    def uniform_iq2(_layer, _attribute):
        return "iq2"

    def mixed_stq(layer, attribute):
        if attribute == DOWN:
            return "iq2"
        return "stq" if layer in low else "iq2"

    def mixed_ltc(layer, attribute):
        if attribute == DOWN:
            return "iq2"
        return "ltc" if layer in low else "iq2"

    catalog = {
        "dense": None,
        "uniform_iq2": uniform_iq2,
        "mixed_stq": mixed_stq,
        "mixed_ltc": mixed_ltc,
    }
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]
    if args.sweep_low_layers:
        for raw in args.sweep_low_layers.split(","):
            count = int(raw.strip())
            band = set(range(count))

            def make(band_set):
                def plan(layer, attribute):
                    if attribute == DOWN:
                        return "iq2"
                    return "ltc" if layer in band_set else "iq2"
                return plan

            name = "ltc_low%d" % count
            catalog[name] = make(band)
            selected.append(name)
    for entry in args.layer_set or []:
        label, _, raw = entry.partition("=")
        band = {int(v) for v in raw.split(",") if v.strip()}

        def make_explicit(band_set):
            def plan(layer, attribute):
                if attribute == DOWN:
                    return "iq2"
                return "ltc" if layer in band_set else "iq2"
            return plan

        catalog[label] = make_explicit(band)
        selected.append(label)
    arms = tuple((name, catalog[name]) for name in selected)

    results = {}
    reference = None
    reference_routing = None
    for arm, plan in arms:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16, device_map=device)
        model.eval()
        if plan is not None:
            stats = apply_plan(model, importance, plan, device)
            results[arm + "_plan"] = stats
            print("%s: bpw %.4f mean_error %.4f" % (arm, stats["bpw"], stats["mean_error"]),
                  flush=True)
        if arm == "dense":
            reference = reference_pass(model, tokenizer, texts, args.seq_len, device)
            reference_routing = router_topk(model, tokenizer, texts, args.seq_len, device)
            total_nll = sum(r["nll_sum"] for r in reference)
            total_tokens = sum(r["tokens"] for r in reference)
            results["dense"] = {
                "nll_per_token": total_nll / total_tokens,
                "perplexity": float(torch.exp(torch.tensor(total_nll / total_tokens))),
                "logit_kl": 0.0, "top1_agreement": 1.0, "margin_ratio": 1.0,
                "tokens": total_tokens, "router_topk_jaccard": 1.0,
            }
            print("dense ppl %.4f" % results["dense"]["perplexity"], flush=True)
        else:
            metrics = measure(model, tokenizer, reference, device)
            metrics["router_topk_jaccard"] = jaccard(
                reference_routing, router_topk(model, tokenizer, texts, args.seq_len, device))
            results[arm] = metrics
            print("%s ppl %.2f kl %.4f top1 %.4f margin %.4f router %.4f" % (
                arm, metrics["perplexity"], metrics["logit_kl"], metrics["top1_agreement"],
                metrics["margin_ratio"], metrics["router_topk_jaccard"]), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
