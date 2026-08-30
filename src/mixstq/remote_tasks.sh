#!/usr/bin/env bash
set -euo pipefail

MODEL="${MIXSTQ_MODEL:-allenai/OLMoE-1B-7B-0924}"
REVISION="${MIXSTQ_REVISION:?set MIXSTQ_REVISION to a pinned commit sha}"
MMLU="${MIXSTQ_MMLU:-140}"
ARC="${MIXSTQ_ARC:-60}"
LOW_LAYERS="${MIXSTQ_LOW_LAYERS:-6}"
ARMS="${MIXSTQ_ARMS:-dense,mixed_stq,mixed_ltc}"
WORKDIR="${MIXSTQ_WORKDIR:-/workspace/mixstq}"

echo "[remote] python: $(python3 --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$WORKDIR/artifacts"
cd "$WORKDIR"

python3 -m pip install -q --upgrade pip
python3 -m pip install -q "transformers>=4.44,<5" "datasets>=3.0" "accelerate>=1.0"

echo "[remote] task accuracy for $MODEL @ $REVISION"
python3 eval_tasks.py \
  --model "$MODEL" \
  --revision "$REVISION" \
  --imatrix artifacts/imatrix.json \
  --mmlu "$MMLU" \
  --arc "$ARC" \
  --low-layers "$LOW_LAYERS" \
  --arms "$ARMS" \
  --out artifacts/task_accuracy.json

echo "[remote] done"
ls -la artifacts/
