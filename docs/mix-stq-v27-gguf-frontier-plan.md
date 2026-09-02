# Qwen3.8-27B GGUF Frontier Implementation Plan

> **2026-08-31 상태:** 유료 실행 전 기반 코드와 pinned GGUF command planner까지 완료했다.
> 실제 v0.27 GGUF 생성·평가는 아직 시작하지 않았다. 현재 상태와 다음 작업은
> `docs/mix-stq-v27-status-and-next-steps.md`를 기준으로 한다.

**Goal:** 고정한 Qwen3.8-27B와 llama.cpp에서 BF16, IQ3_XXS, IQ4_XS, Q4_K_M,
Q5_K_M 실제 GGUF를 만들고 동일 평가·HF 보존·Terminal Bench 2.1 full까지 재현 가능한
증거 묶음으로 완성한다.

**Architecture:** 로컬 코드는 유료 실행 전에 host selection, calibration, run command,
artifact audit와 llama-server 평가 계약을 고정한다. Vast.ai는 credential 없는 계산 노드로만
사용하고, 생성 파일은 SHA manifest를 기준으로 trusted local host가 shard 단위로 HF에
보존한다. 각 구현 task는 별도 commit과 독립 spec/quality review를 통과한 뒤 다음 task가
그 interface를 사용한다.

**Tech Stack:** Python 3.11+, Bash, Vast.ai REST API, Hugging Face Hub, pinned
`llama.cpp`, GGUF, `llama-server`, `llama-imatrix`, `llama-quantize`, pytest, Ruff,
Flint Chart 0.5.1.

**Spec:** `docs/mix-stq-v27-gguf-frontier-preregistered.md`

## Global Constraints

- Model is exactly `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- llama.cpp is exactly commit `580e88d8b7dece7099d9b62323521d0254ff3615`.
- Text conversion uses BF16 and `--no-mtp`; vision projector is separate and excluded from text bpw.
- Every quantized arm is created directly from the same BF16 GGUF with the same llama.cpp imatrix; requantization is forbidden.
- Required arms are `IQ3_XXS`, `IQ4_XS`, `Q4_K_M`, and `Q5_K_M`; a partial arm set is never reported as the completed frontier.
- Calibration is exactly 32 pinned source-order records from each of wiki, code, and chat; MMLU and ARC are excluded.
- Primary Vast host is one clean NVIDIA GPU with at least 80 GB VRAM, 96 decimal GB system RAM, 16 `cpu_cores`, 300 GB disk, 500 Mbps download, 0.98 reliability, and compute price at most $1.20/hour.
- The 800-item evaluation must reproduce fingerprint `a72515282c6fc20f34188b3102d99468ab2b02266105ed9c6e4ec405fbad8fd0` before the first request.
- Hugging Face and Vast credentials never enter repository files, rented-host files, command arguments, detached logs, or public artifacts.
- Quantized artifacts are public-re-download hash verified before the Vast instance is destroyed.
- Terminal Bench is full, free-form, and uses the same task set, agent scaffold, prompt policy, timeout, and concurrency for every arm.
- Internal orchestration artifacts remain gitignored and absent from the public tree.
- Fresh focused tests, `python3 scripts/run_tests.py --offline`, Ruff, `git diff --check`, CLI happy/bad/help checks, and an independent review are required before each paid or external side effect.

---

### Task 1: Validate and publish the paid-run preflight infrastructure

**Files:**
- Verify: `src/mixstq/vast_control.py`
- Verify: `tests/test_vast_control.py`
- Verify: `src/mixstq/llama_calibration.py`
- Verify: `tests/test_llama_calibration.py`

**Interfaces:**
- Produces: Vast search/create constraints `min_system_ram_gb`, `min_cpu_cores`,
  `min_download_mbps`, `min_reliability` with neutral defaults.
- Produces: `python -m mixstq.llama_calibration --out PATH --manifest PATH` with pinned,
  deterministic 96-record output and atomic no-overwrite publication.
- Consumes: no v0.27 result data.

- [ ] **Step 1: Run focused tests from the exact candidate HEAD**

Run:

```bash
python3 -m pytest -q tests/test_vast_control.py tests/test_llama_calibration.py
```

Expected at the current implementation: 113 focused tests pass (91 Vast controller + 22 calibration)
and no test is skipped.

- [ ] **Step 2: Run repository gates**

Run:

```bash
python3 scripts/run_tests.py --offline
python3 -m ruff check src/mixstq/vast_control.py src/mixstq/llama_calibration.py tests/test_vast_control.py tests/test_llama_calibration.py
python3 -m compileall -q src/mixstq/vast_control.py src/mixstq/llama_calibration.py
git diff --check origin/main...HEAD
```

Expected: offline suite reports 11/11, Ruff exits 0, compileall exits 0, and diff check is empty.

- [ ] **Step 3: Manually exercise both CLIs without side effects**

Run:

```bash
PYTHONPATH=src python3 -m mixstq.llama_calibration --help
PYTHONPATH=src python3 -m mixstq.vast_control create --offer 1 --max-hourly 1.20 --disk 300 --min-vram 80 --min-system-ram-gb 96 --min-cpu-cores 16 --min-download-mbps 500 --min-reliability 0.98
```

Expected: calibration help lists `--out`, `--manifest`, `--per-domain`, `--min-chars`;
Vast command prints `DRY RUN` and makes no API create request.

- [ ] **Step 4: Independent task review and push**

Review the two existing commit ranges separately, require both spec compliance and quality approval,
then push the immutable reviewed HEAD to `origin/main`. Confirm local and remote 40-character SHA are equal.

---

### Task 2: Add the pinned Vast GGUF runbook and command planner

**Files:**
- Create: `docs/mix-stq-v27-vast-runbook.md`
- Create: `src/mixstq/gguf_run_plan.py`
- Create: `tests/test_gguf_run_plan.py`

**Interfaces:**
- Produces: `gguf_run_plan.build_plan(workspace: Path, run_commit: str) -> dict[str, list[list[str]]]`.
- Produces: `python -m mixstq.gguf_run_plan --workspace /workspace --run-commit SHA --format shell|json`.
- Produces phases `bootstrap`, `calibration`, `convert`, `imatrix`, `quantize`, `smoke`,
  `audit`, `split`, and `upload`; command arguments contain no credential.
- Consumes: calibration CLI and exact constants from the v0.27 spec.

- [ ] **Step 1: Write failing contract tests**

Create tests that call `build_plan(Path("/workspace"), "a" * 40)` and assert:

```python
assert list(plan) == [
    "bootstrap", "calibration", "convert", "imatrix",
    "quantize", "smoke", "audit", "split",
]
assert "580e88d8b7dece7099d9b62323521d0254ff3615" in flattened
assert "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in flattened
assert "--no-mtp" in flattened
for tier in ("IQ3_XXS", "IQ4_XS", "Q4_K_M", "Q5_K_M"):
    assert tier in flattened
for forbidden in ("hf_", "MIXSTQ_VAST_KEY", "HF_TOKEN", "--token"):
    assert forbidden not in flattened
```

Also assert a non-40-hex run commit raises `ValueError`, all output files live below
`/workspace/mix-stq/artifacts/qwen38-gguf-v27`, and every quantizer command consumes the same
BF16 path and imatrix path.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q tests/test_gguf_run_plan.py
```

Expected: import failure because `mixstq.gguf_run_plan` does not exist.

- [ ] **Step 3: Implement immutable command construction**

Use tuple constants and argv lists rather than interpolated shell text:

```python
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LLAMA_COMMIT = "580e88d8b7dece7099d9b62323521d0254ff3615"
TIERS = ("IQ3_XXS", "IQ4_XS", "Q4_K_M", "Q5_K_M")

def build_plan(workspace: Path, run_commit: str) -> dict[str, list[list[str]]]:
    if re.fullmatch(r"[0-9a-f]{40}", run_commit) is None:
        raise ValueError("run commit must be 40 lowercase hex characters")
    root = workspace / "mix-stq"
    artifacts = root / "artifacts" / "qwen38-gguf-v27"
    bf16 = artifacts / "qwen38-27b-bf16.gguf"
    imatrix = artifacts / "qwen38-27b-imatrix.gguf"
    return ordered_phase_commands(root, artifacts, bf16, imatrix, run_commit)
```

`ordered_phase_commands` must emit exact detached git fetch/checkouts, CUDA CMake release build,
pinned HF snapshot download, calibration command, converter commands, one imatrix command,
four direct quantizer commands, five independent smoke commands, audit commands, and 8 GiB split commands.

- [ ] **Step 4: Write the runbook from the same interface**

The runbook must include:

- local reviewed SHA injection and detached remote checkout
- GPU/RAM/disk/process preflight and stop conditions
- `tmux` session names and per-phase logs/completion markers
- command planner JSON capture before execution
- phase-by-phase command execution with exact output paths
- one-arm-at-a-time shard recovery and public HF re-download verification
- destruction only after manifest closure

No secret value or example token is permitted.

- [ ] **Step 5: Run GREEN and CLI manual QA**

Run:

```bash
python3 -m pytest -q tests/test_gguf_run_plan.py
PYTHONPATH=src python3 -m mixstq.gguf_run_plan --help
PYTHONPATH=src python3 -m mixstq.gguf_run_plan --workspace /workspace --run-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --format json
PYTHONPATH=src python3 -m mixstq.gguf_run_plan --workspace /workspace --run-commit invalid --format json
```

Expected: tests pass; help exits 0; valid JSON contains all eight ordered phases; invalid SHA exits nonzero
without creating files.

- [ ] **Step 6: Commit**

```bash
git add docs/mix-stq-v27-vast-runbook.md src/mixstq/gguf_run_plan.py tests/test_gguf_run_plan.py
git commit -m "Add the pinned Qwen GGUF paid-run plan"
```

---

### Task 3: Add a GGUF physical-bpw and provenance auditor

**Files:**
- Create: `src/mixstq/gguf_audit.py`
- Create: `tests/test_gguf_audit.py`

**Interfaces:**
- Produces: `summarize_gguf(path: Path, expected_arch: str = "qwen35") -> dict[str, object]`.
- Produces JSON fields `sha256`, `file_bytes`, `architecture`, `file_type`, `tensor_count`,
  `tensor_elements`, `tensor_payload_bytes`, `physical_bpw`, `payload_bpw`, and `tensors_by_type`.
- Consumes: `gguf.GGUFReader` from the pinned llama.cpp checkout at runtime; tests inject a fake reader.

- [ ] **Step 1: Write failing arithmetic and validation tests**

Use a fake reader with two tensors totaling 1,000 elements and 400 payload bytes. For a 500-byte
file assert `physical_bpw == 4.0` and `payload_bpw == 3.2`. Assert wrong architecture,
zero elements, overlapping tensor spans, non-finite bpw, and missing file type each raise an explicit error.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_gguf_audit.py
```

Expected: import failure for `mixstq.gguf_audit`.

- [ ] **Step 3: Implement the pure summary core and thin GGUF adapter**

The arithmetic core takes already-normalized tensor records:

```python
def summarize_records(file_bytes: int, metadata: Mapping[str, object], tensors: Sequence[TensorRecord]) -> dict[str, object]:
    tensor_elements = sum(record.elements for record in tensors)
    tensor_payload_bytes = sum(record.nbytes for record in tensors)
    if tensor_elements <= 0:
        raise ValueError("GGUF has no tensor elements")
    return {
        "file_bytes": file_bytes,
        "tensor_count": len(tensors),
        "tensor_elements": tensor_elements,
        "tensor_payload_bytes": tensor_payload_bytes,
        "physical_bpw": file_bytes * 8.0 / tensor_elements,
        "payload_bpw": tensor_payload_bytes * 8.0 / tensor_elements,
    }
```

The adapter resolves `general.architecture`, `general.file_type`, tensor shape/type/data span and adds
the streaming SHA-256. It must not load tensor payloads into RAM.

- [ ] **Step 4: Run GREEN and bad-input CLI QA**

```bash
python3 -m pytest -q tests/test_gguf_audit.py
PYTHONPATH=src python3 -m mixstq.gguf_audit --help
PYTHONPATH=src python3 -m mixstq.gguf_audit --model /does/not/exist.gguf --out /tmp/unused.json
```

Expected: focused tests pass, help exits 0, missing input exits nonzero and creates no output.

- [ ] **Step 5: Commit**

```bash
git add src/mixstq/gguf_audit.py tests/test_gguf_audit.py
git commit -m "Add GGUF physical bit audit"
```

---

### Task 4: Add the strict llama-server 800-item evaluator

**Files:**
- Create: `src/mixstq/eval_llama_server.py`
- Create: `tests/test_eval_llama_server.py`

**Interfaces:**
- Consumes: `eval_tasks.load_mmlu_stratified(10)`, `eval_tasks.load_arc(230)`,
  `eval_tasks.validate_item_counts`, `eval_tasks.item_fingerprint`, and `eval_tasks.render`.
- Consumes HTTP: `POST /tokenize` and `POST /completion` from the pinned llama-server.
- Produces: immutable arm JSON with provenance, 800 item records, correctness vector, timing,
  completion token ids, and an atomic completion marker.
- Produces CLI: `python -m mixstq.eval_llama_server --server URL --arm NAME --model-sha256 SHA --llama-commit SHA --out PATH`.

- [ ] **Step 1: Write failing HTTP-contract tests**

Use an in-process fake HTTP server. Cover:

```python
assert request["n_predict"] == 1
assert request["seed"] == 22
assert request["cache_prompt"] is False
assert request["repeat_penalty"] == 1.0
assert sorted(bias for _, bias in request["logit_bias"]) == [100.0] * 4
```

Also cover four distinct single-token letters, multi-token letter rejection, candidate-external output
rejection, top-4 candidate-set rejection, wrong 800 fingerprint rejection before completion requests,
resume provenance mismatch, partial output not promoted, and refusal to overwrite a completed result.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_eval_llama_server.py
```

Expected: import failure for `mixstq.eval_llama_server`.

- [ ] **Step 3: Implement strict request and publication flow**

Build each request with an explicit mapping:

```python
payload = {
    "prompt": prompt,
    "n_predict": 1,
    "temperature": -1.0,
    "seed": 22,
    "cache_prompt": False,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "logit_bias": [[token_id, 100.0] for token_id in letter_ids],
    "n_probs": 4,
    "return_tokens": True,
}
```

Before scoring, load exactly 570+230 items and compare the ordered fingerprint to the fixed value.
After every item atomically write progress. Only after all 800 records pass schema/provenance checks write
the immutable result and completion marker.

- [ ] **Step 4: Run GREEN and CLI QA**

```bash
python3 -m pytest -q tests/test_eval_llama_server.py
PYTHONPATH=src python3 -m mixstq.eval_llama_server --help
PYTHONPATH=src python3 -m mixstq.eval_llama_server --server http://127.0.0.1:1 --arm BF16 --model-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --llama-commit 580e88d8b7dece7099d9b62323521d0254ff3615 --out /tmp/qwen38-no-server.json
```

Expected: tests pass, help exits 0, unreachable server exits nonzero and does not create a completion marker.

- [ ] **Step 5: Commit**

```bash
git add src/mixstq/eval_llama_server.py tests/test_eval_llama_server.py
git commit -m "Add strict llama server accuracy evaluation"
```

---

### Task 5: Review, provision Vast, and generate the five GGUF arms

**Files:**
- Produce remotely: `artifacts/qwen38-gguf-v27/**`
- Record locally: hidden SDD ledger and immutable run manifest only; no credential files are copied.

**Interfaces:**
- Consumes: reviewed/pushed commit from Tasks 1–4 and `gguf_run_plan` JSON.
- Produces: BF16, IQ3_XXS, IQ4_XS, Q4_K_M, Q5_K_M monolithic GGUF files, one imatrix,
  one projector, calibration corpus/manifest, per-phase logs, completion markers, and audits.
- The `convert` phase emits two commands: the text BF16 GGUF (`--no-mtp`) and, from the same
  pinned converter, the vision projector (`--mmproj`), which stays out of every text bpw
  denominator and out of the benchmarks.
- `bootstrap` builds and probes eight executables, `llama-server`, `llama-perplexity` and
  `llama-bench` included, so Task 6 needs no extra build. Each probe requires usage text and
  accepts the usage exit codes, because `llama-quantize --help` exits 1 at the pinned commit.

- [ ] **Step 1: Run final pre-rental gates**

```bash
python3 scripts/run_tests.py --offline
python3 -m pytest -q tests/test_vast_control.py tests/test_llama_calibration.py tests/test_gguf_run_plan.py tests/test_gguf_audit.py tests/test_eval_llama_server.py
python3 -m ruff check .
git diff --check
test -z "$(git status --porcelain)"
```

- [ ] **Step 2: Search and revalidate an 80 GB offer**

```bash
PYTHONPATH=src python3 -m mixstq.vast_control search --gpu '' --max-price 1.20 --min-vram 80 --disk 300 --limit 20 --min-system-ram-gb 96 --min-cpu-cores 16 --min-download-mbps 500 --min-reliability 0.98
```

Pass the same constraints to `create --confirm`. Vast hands out a different chunk id for the same
machine on every `/bundles` call, so the id from this search is only a preference: `create --confirm`
re-searches, re-checks every numeric constraint against the fresh offer, and rents the id that same
response returned. Add `--exclude-machine <ID>` for any machine already observed to fail (machine
`142444` could not attach a GPU to a container, twice), and `--machine <ID>` to pin a known-good one.

- [ ] **Step 3: Execute phases in one tmux session**

Record start/end epoch, host offer, runtime versions, GPU inventory and planner JSON. Run each phase in
order and require its completion marker before the next. Do not change the model revision, llama commit,
corpus, imatrix settings or tiers in response to intermediate output.

- [ ] **Step 4: Audit and smoke every arm**

Require five exit-0 `llama-cli` logs, five audit JSON files, unique model SHA values, identical recorded
BF16 input/imatrix SHA for quantized arms, and no unsupported/NaN/abort marker.

---

### Task 6: Preserve artifacts and run Top-1, PPL, and llama-bench

**Files:**
- Produce: arm correctness JSON, paired comparison JSON, PPL logs, llama-bench JSON/Markdown,
  peak memory logs, split manifests, HF preservation marker.

**Interfaces:**
- Consumes: only smoke-approved GGUF files from Task 5.
- Produces: complete metrics for all five arms and public HF paths for all four quantized arms.

- [ ] **Step 1: Run each arm in a fresh llama-server process**

For each model, wait for `/health`, run the strict 800-item evaluator, validate its completion marker,
then terminate the server and confirm no GPU process remains before starting the next arm.

- [ ] **Step 2: Compute paired statistics**

Assemble arm correctness vectors in the fixed order and call `task_accuracy.compare` with `BF16` as
baseline. Add the preregistered 2%p decision for each quantized arm without changing the generic statistic.

- [ ] **Step 3: Run held-out PPL and llama-bench**

Use the pinned WikiText-2 test corpus, identical context/chunk settings, full GPU offload, and
`llama-bench` prompt 512/generation 128/repetitions 5 for every arm. Preserve raw repetitions and medians.

- [ ] **Step 4: Split, upload from the host, and public-verify one arm at a time**

Split quantized files to at most 8 GiB, then run the planner's `upload` phase. Per amendment 3 the
rented host uploads directly to `topabaem/mix-stq-artifacts` under
`paid-run/qwen38-gguf-frontier-v27/<ARM>/`, plus `projector/` and `evidence/`, using the CLI the
user logged in by hand; no token appears in argv, logs or planner output. Each upload is followed
by an unauthenticated public re-download in a process with the Hub and token environment stripped,
and a sha256 comparison that releases each verified copy immediately, so the verification never
needs more than one shard of extra disk. Keep every remote monolith until all four arms have a
valid public preservation marker. The BF16 monolith is not uploaded; its revision and SHA are the
preservation record.

- [ ] **Step 5: Destroy Vast only after closure**

Recover logs/manifests/results, verify aggregate hashes, destroy the instance, and require final Vast list
`[]` and burn `$0.0000`.

---

### Task 7: Publish the five-stage result report and Flint Chart

**Files:**
- Create: `docs/mix-stq-v27-gguf-frontier-results.md`
- Modify: `README.md`
- Modify: `docs/figs/build_qwen38.mjs`
- Modify: `docs/figs/qwen38_top1.json`
- Modify: `docs/figs/qwen38_top1.svg`
- Modify: `docs/figs/qwen38_top1.png`

**Interfaces:**
- Consumes: only immutable manifests/results from Task 6.
- Produces: a five-stage Korean report and one Flint Chart showing actual GGUF Top-1 arms together,
  with v0.26 MLP-only evidence visually and textually separated.

- [ ] **Step 1: Bind every table cell to artifact evidence**

Report model/llama SHA, physical and payload bpw, file sizes, Top-1 counts, BF16 deltas/CIs/McNemar,
PPL, prompt/generation throughput, peak VRAM, wall time, cost and HF URLs. Mark missing values as
`미측정`; never substitute literature or model-card values into measured columns.

- [ ] **Step 2: Render the chart with Flint 0.5.1**

Use one mark/axis for actual GGUF arms, labels to three decimals, and a separate annotation for the
v0.26 MLP-only point. Generate JSON, SVG, and PNG from the same source data.

- [ ] **Step 3: Verify report/chart consistency and commit**

Run node syntax, reproducible chart render, exact JSON-to-report value checks, image inspection,
offline suite, Ruff and diff check, then obtain an independent final review before push.

---

### Task 8: Run free-form Terminal Bench 2.1 full on the target server

**Files:**
- Produce on `topabaem@100.73.38.99`: per-arm Terminus run directories and immutable manifests.
- Add after recovery: `docs/mix-stq-v28-terminal-bench-results.md`.

**Interfaces:**
- Consumes: HF-public-verified quantized GGUF shards and the exact pinned llama.cpp runtime.
- Produces: task-level reward/pass/fail/error/timeout and wall time for every full arm run.

- [ ] **Step 1: Audit target server before transfer**

Record OS, free disk/RAM, GPU, existing workloads, Claude Code and Terminal Bench versions. Do not stop
unrelated processes. Select concurrency that fits observed resources and freeze it for every arm.

- [ ] **Step 2: Transfer and verify the GGUFs**

Download from the public HF preservation path or transfer verified shards. Reassemble/load through the
first shard as required by llama.cpp and compare aggregate manifest hashes before serving.

- [ ] **Step 3: Freeze the free-form harness**

Use the same Claude Code wrapper, system prompt, tools, timeout, task set and concurrency for all arms.
Do not force answer choices, regex-repair answers, copy another arm's answer, or selectively rerun tasks.

- [ ] **Step 4: Run every arm full and recover evidence**

Run IQ3_XXS, IQ4_XS, Q4_K_M and Q5_K_M as separate full jobs. A failed full job is preserved as failed;
its replacement is a new whole-arm run, not a stitched subset.

- [ ] **Step 5: Report measured scores and times**

Publish task-level and aggregate results, paired task outcomes where identities align, error taxonomy,
wall time and exact harness/runtime/model identities. Distinguish Terminal Bench evidence from 800-item
Top-1 and from model-card BF16 numbers.
