# MIX-STQ v0.26 Vast.ai paid-run runbook

This runbook is for one clean Vast.ai instance with one 96 GB NVIDIA GPU. Stop on any failed
command. Do not destroy the instance until both the Hugging Face upload and local hash check pass.

## 0. Pre-rental gate

Do not create or rent a Vast instance until all of the following are true:

- `MIXSTQ_RUN_COMMIT` has been supplied externally as the reviewed immutable 40-character
  lowercase hexadecimal commit SHA; never derive it from a branch such as `origin/main`.
- The reviewed SHA contains this paid-run gate wave and its focused and full offline tests pass.
- The exact model `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and protocol `qwen38_bf16_800` remain approved.

Vast and Hugging Face credentials stay on the trusted local machine. Never place a Hugging Face
write token or token file on the rented host. Never place API or Hugging Face tokens in command
arguments, shell history, detached logs, artifacts, or this file.

## 1. Clean instance and GPU preflight

Run on the new instance before cloning or loading any model artifact:

```bash
set -euo pipefail
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
df -h /workspace
```

The inventory must show exactly one GPU, at least `92160 MiB` total memory, and no compute
process. The evaluator repeats this check and fails closed if CUDA, `nvidia-smi`, parsing, memory,
or exclusivity is wrong.

## 2. Immutable checkout and pinned environment

```bash
set -euo pipefail
cd /workspace
: "${MIXSTQ_RUN_COMMIT:?supply the externally reviewed immutable commit SHA}"
test "${#MIXSTQ_RUN_COMMIT}" -eq 40
case "$MIXSTQ_RUN_COMMIT" in
  *[!0-9a-f]*) echo "MIXSTQ_RUN_COMMIT must be 40 lowercase hex characters" >&2; exit 1 ;;
esac
git clone --no-checkout https://github.com/Topabaem05/mix-stq.git
cd mix-stq
git fetch --force origin "$MIXSTQ_RUN_COMMIT"
test "$(git rev-parse FETCH_HEAD)" = "$MIXSTQ_RUN_COMMIT"
git checkout --detach "$MIXSTQ_RUN_COMMIT"
test "$(git rev-parse HEAD)" = "$MIXSTQ_RUN_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p artifacts
printf '%s\n' "$MIXSTQ_RUN_COMMIT" | tee artifacts/run-commit.txt

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
python3 -m pip install transformers==5.16.1 datasets==4.8.5 accelerate==1.14.0
python3 -m pip install --no-deps -e .
python3 -m pip check
python3 - <<'PY'
import datasets
import accelerate
import torch
import torchvision
import transformers

assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torchvision.__version__ == "0.26.0+cu128", torchvision.__version__
assert transformers.__version__ == "5.16.1", transformers.__version__
assert datasets.__version__ == "4.8.5", datasets.__version__
assert accelerate.__version__ == "1.14.0", accelerate.__version__
assert torch.cuda.is_available()
print(
    torch.__version__, torchvision.__version__, transformers.__version__, datasets.__version__,
    accelerate.__version__,
)
PY
```

`MIXSTQ_RUN_COMMIT` is the externally reviewed immutable source identity for this run. Record it
and do not pull, edit, or change environments after this point.

## 3. Transfer and verify the importance tensor

Do not configure Hugging Face authentication on the rented host. Download the public importance
tensor without credentials. If public download is unavailable, transfer it from the trusted local
machine with `scp`; do not solve that failure by copying a token to the host.

```bash
set -euo pipefail
cd /workspace/mix-stq
source .venv/bin/activate
curl --fail --location --silent --show-error \
  --output artifacts/qwen38_imatrix.pt \
  https://huggingface.co/datasets/topabaem/mix-stq-artifacts/resolve/main/qwen38_imatrix.pt
test "$(stat -c %s artifacts/qwen38_imatrix.pt)" = "7137641"
printf '%s  %s\n' \
  def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0 \
  artifacts/qwen38_imatrix.pt | sha256sum -c -
```

If transfer is done with `scp`, place the file at the same path and run the same size and SHA-256
checks before continuing. Do not run `hf auth login` or place any Hugging Face token on the host.

## 4. Detached paid run

```bash
set -euo pipefail
cd /workspace/mix-stq
source .venv/bin/activate
nohup env PYTHONPATH=src/mixstq python3 -u src/mixstq/eval_tasks.py \
  --protocol qwen38_bf16_800 \
  --model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --imatrix artifacts/qwen38_imatrix.pt \
  --mmlu-per-subject 10 \
  --arc 230 \
  --arms dense,dense_iq3_ref \
  --dtype bfloat16 \
  --out artifacts/qwen38_bf16_800.json \
  > artifacts/qwen38_bf16_800.log 2>&1 &
printf '%s\n' "$!" > artifacts/qwen38_bf16_800.pid
tail -F artifacts/qwen38_bf16_800.log
```

The lifecycle files are:

- progress: `artifacts/qwen38_bf16_800.json.progress.json`
- success marker: `artifacts/qwen38_bf16_800.json.complete.json`
- failure: `artifacts/qwen38_bf16_800.json.failure.json`
- same-instance arm caches: `artifacts/correct_dense.json` and
  `artifacts/correct_dense_iq3_ref.json`

For same-instance recovery after a failure, inspect the failure and progress JSON, correct the
environmental cause without changing code/config/data, and rerun the exact command above. Valid
current-schema caches are reused; stale or mismatched strict caches are quarantined. Never treat the
result JSON as complete without validating its completion marker. Once either the result or
completion marker exists, the strict evaluator refuses to rerun at that output path and preserves
the existing artifact; preserve it rather than deleting or quarantining it.

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

result = Path("artifacts/qwen38_bf16_800.json")
marker = Path("artifacts/qwen38_bf16_800.json.complete.json")
payload = json.loads(marker.read_text(encoding="utf-8"))
actual = hashlib.sha256(result.read_bytes()).hexdigest()
assert payload["status"] == "complete"
assert payload["result_sha256"] == actual
print(actual)
PY
```

## 5. Hash and preserve before destroy

On the rented instance, only create and verify the GNU `sha256sum` manifest. Do not authenticate to
Hugging Face and do not upload from the host:

```bash
set -euo pipefail
cd /workspace/mix-stq
find artifacts -maxdepth 1 -type f ! -name qwen38_bf16_800.manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > artifacts/qwen38_bf16_800.manifest.sha256
sha256sum -c artifacts/qwen38_bf16_800.manifest.sha256
```

From the trusted local machine, copy the directory and verify every manifest entry. This writes a
local-recovery marker containing the verified manifest digest:

```bash
set -euo pipefail
PRESERVATION_DIR="$(pwd)/mix-stq-paid-run"
mkdir -p "$PRESERVATION_DIR"
scp -P "$VAST_SSH_PORT" -r \
  "root@$VAST_SSH_HOST:/workspace/mix-stq/artifacts" "$PRESERVATION_DIR/"
cd "$PRESERVATION_DIR"
python3 - <<'PY'
import hashlib
from pathlib import Path

manifest = Path("artifacts/qwen38_bf16_800.manifest.sha256")
for line in manifest.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split(maxsplit=1)
    path = Path(relative.lstrip(" *"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("SHA-256 mismatch: %s" % path)
    print("%s: OK" % path)
manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
Path(".local-recovery-verified").write_text(manifest_digest + "\n", encoding="utf-8")
print("local recovery verified: %s" % manifest_digest)
PY
```

Only after local verification succeeds, upload from the trusted local machine using its existing
Hugging Face credential store/token file. Never put the token in arguments, environment exports,
shell history, or logs. Then download the remote directory into a new temporary directory and
verify the remote manifest and every remote artifact against the local verified manifest:

```bash
set -euo pipefail
cd "$PRESERVATION_DIR"
test -s .local-recovery-verified
hf auth whoami
hf upload topabaem/mix-stq-artifacts artifacts/ paid-run/qwen38-bf16-800/ \
  --repo-type dataset
HF_VERIFY_DIR="$(mktemp -d)"
export HF_VERIFY_DIR
trap 'rm -rf "$HF_VERIFY_DIR"' EXIT
hf download topabaem/mix-stq-artifacts \
  --repo-type dataset \
  --include 'paid-run/qwen38-bf16-800/*' \
  --local-dir "$HF_VERIFY_DIR"
python3 - <<'PY'
import hashlib
import os
from pathlib import Path

local_manifest = Path("artifacts/qwen38_bf16_800.manifest.sha256")
remote_root = Path(os.environ["HF_VERIFY_DIR"]) / "paid-run/qwen38-bf16-800"
remote_manifest = remote_root / local_manifest.name
if remote_manifest.read_bytes() != local_manifest.read_bytes():
    raise SystemExit("remote manifest differs from verified local manifest")
for line in local_manifest.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split(maxsplit=1)
    local_path = Path(relative.lstrip(" *"))
    remote_path = remote_root / local_path.name
    actual = hashlib.sha256(remote_path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("remote SHA-256 mismatch: %s" % remote_path)
    print("%s: REMOTE OK" % remote_path)
manifest_digest = hashlib.sha256(local_manifest.read_bytes()).hexdigest()
Path(".hf-preservation-verified").write_text(manifest_digest + "\n", encoding="utf-8")
print("Hugging Face preservation verified: %s" % manifest_digest)
PY
```

Do not continue unless both marker files exist and contain the same manifest digest. Local recovery
without verified Hugging Face preservation, or upload without verified local recovery, is not
sufficient to destroy the instance.

## 6. Destroy and verify empty

From the trusted local checkout with the Vast credential already configured:

```bash
set -euo pipefail
: "${PRESERVATION_DIR:?set this to the absolute trusted-local recovery directory}"
test -s "$PRESERVATION_DIR/.local-recovery-verified"
test -s "$PRESERVATION_DIR/.hf-preservation-verified"
LOCAL_MANIFEST_SHA="$(cat "$PRESERVATION_DIR/.local-recovery-verified")"
HF_MANIFEST_SHA="$(cat "$PRESERVATION_DIR/.hf-preservation-verified")"
test "${#LOCAL_MANIFEST_SHA}" -eq 64
test "$LOCAL_MANIFEST_SHA" = "$HF_MANIFEST_SHA"
python3 src/mixstq/vast_control.py destroy --id "$VAST_INSTANCE_ID" --confirm
python3 src/mixstq/vast_control.py list
```

The final list output must contain the active-instance array `[]` and total burn `$0.0000` before
the paid-run procedure is considered closed.
