from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_ltc as tl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERT_SUFFIX = "mlp.experts"


def held_out_texts(count, minimum_chars=400):
    stream = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    texts = []
    for row in stream:
        text = (row.get("text") or "").strip()
        if len(text) < minimum_chars:
            continue
        texts.append(text)
        if len(texts) >= count:
            break
    return texts


def gather_reference(model, tokenizer, texts, seq_len, device):
    records = []
    with torch.no_grad():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            ids = batch["input_ids"].to(device)
            out = model(input_ids=ids)
            logits = out.logits[0, :-1].float()
            targets = ids[0, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            records.append({
                "ids": ids.cpu(),
                "log_probs": log_probs.cpu(),
                "top1": logits.argmax(dim=-1).cpu(),
                "nll_sum": float(nll.sum()),
                "tokens": int(targets.numel()),
                "margin": float((logits.topk(2, dim=-1).values[:, 0]
                                 - logits.topk(2, dim=-1).values[:, 1]).mean()),
            })
    return records


def evaluate(model, tokenizer, records, device):
    nll_sum = 0.0
    tokens = 0
    kl_sum = 0.0
    kl_count = 0
    agree = 0
    compared = 0
    margin_ratio = []
    with torch.no_grad():
        for record in records:
            ids = record["ids"].to(device)
            out = model(input_ids=ids)
            logits = out.logits[0, :-1].float()
            targets = ids[0, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            nll_sum += float(nll.sum())
            tokens += int(targets.numel())
            reference = record["log_probs"].to(device)
            kl = torch.nn.functional.kl_div(
                log_probs, reference, log_target=True, reduction="batchmean"
            )
            kl_sum += float(kl)
            kl_count += 1
            top1 = logits.argmax(dim=-1)
            agree += int((top1 == record["top1"].to(device)).sum())
            compared += int(top1.numel())
            top2 = logits.topk(2, dim=-1).values
            margin = float((top2[:, 0] - top2[:, 1]).mean())
            if record["margin"] > 0:
                margin_ratio.append(margin / record["margin"])
    return {
        "nll_per_token": nll_sum / max(tokens, 1),
        "perplexity": float(torch.exp(torch.tensor(nll_sum / max(tokens, 1)))),
        "logit_kl": kl_sum / max(kl_count, 1),
        "top1_agreement": agree / max(compared, 1),
        "margin_ratio": sum(margin_ratio) / max(len(margin_ratio), 1),
        "tokens": tokens,
    }


def router_signature(model, tokenizer, texts, seq_len, device, limit=8):
    signatures = {}
    handles = []

    def hook(name):
        def inner(_module, _inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            probs = torch.softmax(logits, dim=-1)
            top = probs.topk(min(8, probs.shape[-1]), dim=-1).indices
            signatures.setdefault(name, []).append(top.cpu())
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
    return {name: torch.cat(chunks, dim=0) for name, chunks in signatures.items()}


def routing_drift(reference, candidate):
    jaccard = []
    for name, ref in reference.items():
        cand = candidate.get(name)
        if cand is None or cand.shape != ref.shape:
            continue
        for row in range(0, ref.shape[0], max(ref.shape[0] // 512, 1)):
            a = set(int(v) for v in ref[row])
            b = set(int(v) for v in cand[row])
            jaccard.append(len(a & b) / max(len(a | b), 1))
    return sum(jaccard) / max(len(jaccard), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="paired quality evaluation for LTC vs STQ")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    texts = held_out_texts(args.documents)
    print("held-out documents: %d" % len(texts), flush=True)

    store = torch.load(args.imatrix, map_location="cpu")
    importance = {}
    for name, tensor in store.items():
        if name.endswith(".router_hits"):
            continue
        importance[name] = tensor

    results = {}
    reference_records = None
    reference_routing = None

    for arm in ("dense", "stq", "ltc"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16, device_map=device
        )
        model.eval()

        if arm != "dense":
            learn = arm == "ltc"
            patterns = None if learn else tl.stq_patterns(torch.device(device))
            errors = []
            with torch.no_grad():
                for name, module in model.named_modules():
                    if not name.endswith(EXPERT_SUFFIX):
                        continue
                    channel = importance.get(name)
                    if channel is None:
                        continue
                    channel = channel.to(device)
                    for attribute in ("gate_up_proj", "down_proj"):
                        param = getattr(module, attribute, None)
                        if param is None:
                            continue
                        data = param.data
                        flat = data.reshape(-1, data.shape[-1])
                        width = flat.shape[-1]
                        if channel.numel() < width:
                            reps = width // channel.numel()
                            local = channel.repeat(reps)[:width]
                        else:
                            local = channel[:width]
                        quantized, relative = tl.quantize_rows(
                            flat, local, patterns=patterns, learn=learn
                        )
                        param.data = quantized.reshape(data.shape).to(data.dtype)
                        errors.append(relative)
            print("%s: quantized %d expert tensors, mean relative error %.4f" % (
                arm, len(errors), sum(errors) / max(len(errors), 1)), flush=True)
            results[arm + "_weight_error"] = sum(errors) / max(len(errors), 1)

        if arm == "dense":
            reference_records = gather_reference(model, tokenizer, texts, args.seq_len, device)
            reference_routing = router_signature(model, tokenizer, texts, args.seq_len, device)
            dense_nll = sum(r["nll_sum"] for r in reference_records)
            dense_tokens = sum(r["tokens"] for r in reference_records)
            results["dense"] = {
                "nll_per_token": dense_nll / dense_tokens,
                "perplexity": float(torch.exp(torch.tensor(dense_nll / dense_tokens))),
                "logit_kl": 0.0,
                "top1_agreement": 1.0,
                "margin_ratio": 1.0,
                "tokens": dense_tokens,
            }
            print("dense ppl %.4f over %d tokens" % (
                results["dense"]["perplexity"], dense_tokens), flush=True)
        else:
            metrics = evaluate(model, tokenizer, reference_records, device)
            routing = router_signature(model, tokenizer, texts, args.seq_len, device)
            metrics["router_topk_jaccard"] = routing_drift(reference_routing, routing)
            results[arm] = metrics
            print("%s ppl %.4f kl %.4f top1 %.4f margin %.4f router %.4f" % (
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

