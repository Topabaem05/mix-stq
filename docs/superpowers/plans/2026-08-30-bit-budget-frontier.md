# 비트 예산 프론티어 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 태스크 정확도가 dense 대비 5%p 이내로 유지되는 최저 bpw 지점을 찾고, 그 지점에서만 LTC와 고정 코드북을 비교한다.

**Architecture:** 기존 `torch_iq2.quantize_rows`는 8값 레인 + 256엔트리 그리드에 하드코딩되어 있다. 이를 (그리드, 레인폭) 파라미터화된 단일 인코더로 일반화하고, `ggml-common.h`에서 IQ2_S(1024엔트리, 8값)와 IQ3_XXS(256엔트리, 4값) 그리드를 추출해 티어로 등록한다. 그 위에서 4개 팔 스윕을 돌려 손실 곡선을 얻고, 게이트를 통과한 bpw에서만 600문항 LTC 비교를 실행한다.

**Tech Stack:** Python 3.11, torch 2.5.1+cu121, transformers >= 5.0, datasets, vast.ai (Quadro RTX 8000 48 GB), ruff, `scripts/run_tests.py`

**Spec:** `docs/superpowers/specs/2026-08-30-bit-budget-frontier.md`

## Global Constraints

- `transformers>=5.0` 필수. 4.57에서는 OLMoE `mlp.experts`가 `ModuleList`라 `gate_up_proj`가 None이 되고 전 텐서가 skip되어 양자화가 무음 무효화된다 (v18 실측).
- 모델과 리비전은 항상 고정: `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- 모델 로드는 반드시 `device_map={"": device}` + `low_cpu_mem_usage=True`. 컨테이너 RAM이 15 GB일 수 있고 fp16 모델은 14 GB다.
- 새 주석/독스트링을 추가하지 않는다. 저장소의 기존 스타일은 자기설명적 변수명이다.
- 테스트는 pytest 함수가 아니라 **독립 실행 스크립트**다. `scripts/run_tests.py`가 서브프로세스로 돌린다. pytest는 0개를 수집한다.
- ruff 통과 필수: `python3 -m ruff check .` (`ruff`는 PATH에 없고 `python3 -m ruff`로만 실행된다).
- GPU 대여 전 `nvidia-smi`가 `0 MiB, 0 %`임을 확인한다. 공유 GPU에서 v18은 $0.30을 잃었다.
- 작업 종료 시 `destroy --confirm` 후 `list`가 `[]`를 반환하는 것까지 확인한다.
- 모든 통계 판정은 McNemar와 bootstrap CI **둘 다** 일치할 때만 유의로 기록한다.

---

### Task 1: 인코더를 (그리드, 레인폭)으로 일반화

**Files:**
- Modify: `src/mixstq/torch_iq2.py:9-16` (상수 블록), `:19-24` (grid 로더), `:29-33` (`_solve_chunk` 서명), `:55-88` (`quantize_rows`)
- Create: `tests/test_tier_encoders.py`

**Interfaces:**
- Produces: `torch_iq2.quantize_rows(matrix, importance, tier="iq2_xxs")` — `tier`는 `"iq2_xxs" | "iq2_s" | "iq3_xxs"`. 반환은 기존과 동일한 `(quantized_tensor, relative_error)`.
- Produces: `torch_iq2.TIERS` — `dict[str, dict]`, 각 항목은 `{"table": str, "lane": int, "bpw": float}`.
- Consumes: `src/mixstq/iq2xxs_tables.json` (기존), Task 2가 추가하는 `tier_tables.json`.

- [ ] **Step 1: 실패하는 테스트를 작성한다**

`tests/test_tier_encoders.py`:

```python
from __future__ import annotations

import torch
import torch_iq2 as tq

failures = []

expected_bpw = {"iq2_xxs": 2.0625, "iq2_s": 2.5625, "iq3_xxs": 3.0625}
for tier, bpw in expected_bpw.items():
    if tier not in tq.TIERS:
        failures.append("tier %s not registered" % tier)
        continue
    if abs(tq.TIERS[tier]["bpw"] - bpw) > 1e-9:
        failures.append("tier %s bpw %.4f expected %.4f" % (tier, tq.TIERS[tier]["bpw"], bpw))

expected_lane = {"iq2_xxs": 8, "iq2_s": 8, "iq3_xxs": 4}
for tier, lane in expected_lane.items():
    if tier in tq.TIERS and tq.TIERS[tier]["lane"] != lane:
        failures.append("tier %s lane %d expected %d" % (tier, tq.TIERS[tier]["lane"], lane))

torch.manual_seed(0)
matrix = torch.randn(8, 256)
importance = torch.ones(256)
errors = {}
for tier in expected_bpw:
    if tier not in tq.TIERS:
        continue
    quantized, relative = tq.quantize_rows(matrix, importance, tier=tier)
    errors[tier] = relative
    if quantized.shape != matrix.shape:
        failures.append("tier %s changed shape" % tier)
    if not (0.0 < relative < 1.0):
        failures.append("tier %s relative error %.6f out of range" % (tier, relative))

if len(errors) == 3 and not (errors["iq3_xxs"] < errors["iq2_s"] < errors["iq2_xxs"]):
    failures.append("error must decrease as bits increase: %s" % errors)

if failures:
    for line in failures:
        print("FAIL: " + line)
    raise SystemExit(1)
print("PASS: three tiers registered, bpw exact, error monotone in bits")
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cd work/mix-stq && PYTHONPATH=src/mixstq:tests python3 tests/test_tier_encoders.py`
Expected: FAIL — `AttributeError: module 'torch_iq2' has no attribute 'TIERS'`

- [ ] **Step 3: 티어 레지스트리와 파라미터화된 인코더를 구현한다**

`src/mixstq/torch_iq2.py`의 상수 블록을 교체한다:

```python
QK_BLOCK = 256
SUB_LEVELS = 16
COARSE = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)
LANE_CHUNK = 1048576

TIERS = {
    "iq2_xxs": {"table": "iq2xxs_grid", "lane": 8, "bpw": 2.0625},
    "iq2_s": {"table": "iq2s_grid", "lane": 8, "bpw": 2.5625},
    "iq3_xxs": {"table": "iq3xxs_grid", "lane": 4, "bpw": 3.0625},
}
LANE = TIERS["iq2_xxs"]["lane"]
LANES_PER_BLOCK = QK_BLOCK // LANE
BLOCK_BYTES = 2 + (QK_BLOCK // 8) * 2
IQ2XXS_BPW = TIERS["iq2_xxs"]["bpw"]
```

`grid` 로더를 티어별로 바꾼다:

```python
@lru_cache(maxsize=8)
def grid(device_str: str, tier: str = "iq2_xxs") -> torch.Tensor:
    table_name = TIERS[tier]["table"]
    if table_name == "iq2xxs_grid":
        payload = json.loads(
            (Path(__file__).with_name("iq2xxs_tables.json")).read_text(encoding="utf-8")
        )
        points = payload["grid"]
    else:
        payload = json.loads(
            (Path(__file__).with_name("tier_tables.json")).read_text(encoding="utf-8")
        )
        points = payload[table_name]
    return torch.tensor(points, dtype=torch.float32, device=torch.device(device_str))
```

`_solve_chunk`는 `lanes_per_block`을 인자로 받게 한다:

```python
def _solve_chunk(magnitude, weights, base, table, table_square, lanes_per_block):
    linear = (weights * magnitude) @ table.t()
    quadratic = weights @ table_square.t()
    lane_base = base.repeat_interleave(lanes_per_block, dim=0)
```

이하 본문은 그대로 두고, `quantize_rows` 서명과 레인 계산을 바꾼다:

```python
def quantize_rows(matrix, importance, tier="iq2_xxs"):
    device = matrix.device
    lane = TIERS[tier]["lane"]
    lanes_per_block = QK_BLOCK // lane
    table = grid(str(device), tier)
    table_square = table.square()
    work = matrix.to(torch.float32)
    rows, width = work.shape
    usable = (width // QK_BLOCK) * QK_BLOCK
    if usable == 0:
        return matrix.clone(), 0.0

    body = work[:, :usable].reshape(-1, QK_BLOCK)
    channel = importance.to(torch.float32)[:usable]
    weight_body = channel.reshape(1, usable).expand(rows, usable).reshape(-1, QK_BLOCK)

    recon = torch.empty_like(body)
    block_step = max(LANE_CHUNK // lanes_per_block, 1)
    for start in range(0, body.shape[0], block_step):
        block = body[start : start + block_step]
        block_weight = weight_body[start : start + block_step]
        base = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / table.max()
        lanes = block.reshape(-1, lane)
        lane_weights = block_weight.reshape(-1, lane)
        signs = torch.where(lanes < 0, -1.0, 1.0)
        magnitude = lanes.abs()
        index, step = _solve_chunk(
            magnitude, lane_weights, base, table, table_square, lanes_per_block)
        recon[start : start + block_step] = (
            table[index] * step.unsqueeze(1) * signs
        ).reshape(block.shape)

    total_error = (weight_body * (body - recon).square()).double().sum()
    total_energy = (weight_body * body.square()).double().sum()
    out = work.clone()
    out[:, :usable] = recon.reshape(rows, usable)
    return out.to(matrix.dtype), float(total_error / total_energy.clamp_min(1e-30))
```

기존 `/ 43.0`을 `/ table.max()`로 바꾼 것이 핵심이다. 43은 IQ2 그리드의 최대 크기값이고, IQ3_XXS는 62다.

- [ ] **Step 4: Task 2의 테이블이 아직 없으므로 iq2_xxs만 통과함을 확인한다**

Run: `cd work/mix-stq && PYTHONPATH=src/mixstq:tests python3 tests/test_tier_encoders.py`
Expected: FAIL — `FileNotFoundError: tier_tables.json`. Task 2에서 해소된다.

- [ ] **Step 5: 커밋**

```bash
cd work/mix-stq
git add src/mixstq/torch_iq2.py tests/test_tier_encoders.py
git commit -m "Parameterize the codebook encoder by grid and lane width"
```

---

### Task 2: IQ2_S와 IQ3_XXS 그리드를 ggml 헤더에서 추출

**Files:**
- Modify: `src/mixstq/extract_iq2_tables.py`
- Create: `src/mixstq/tier_tables.json` (스크립트 산출물)
- Test: `tests/test_tier_encoders.py` (Task 1에서 작성됨, 여기서 통과시킨다)

**Interfaces:**
- Consumes: `csrc/ggml-common.h`의 `iq2s_grid` (uint64_t, 1024엔트리), `iq3xxs_grid` (uint32_t, 256엔트리)
- Produces: `src/mixstq/tier_tables.json` — `{"iq2s_grid": [[int]*8]*1024, "iq3xxs_grid": [[int]*4]*256}`

- [ ] **Step 1: 추출기에 다중 타입 지원을 추가한다**

`src/mixstq/extract_iq2_tables.py` 끝에 덧붙인다:

```python
def unpack(name: str, bytes_per_entry: int) -> list[list[int]]:
    values = table(name)
    return [
        [(packed >> (8 * j)) & 0xFF for j in range(bytes_per_entry)]
        for packed in values
    ]


tier_payload = {
    "iq2s_grid": unpack("iq2s_grid", 8),
    "iq3xxs_grid": unpack("iq3xxs_grid", 4),
}
for name, points in tier_payload.items():
    magnitudes = sorted({v for entry in points for v in entry})
    print("%s: %d entries x %d values, magnitudes %s" % (
        name, len(points), len(points[0]), magnitudes))
Path("tier_tables.json").write_text(json.dumps(tier_payload), encoding="utf-8")
print("wrote tier_tables.json")
```

- [ ] **Step 2: 헤더를 옆에 두고 추출기를 실행한다**

```bash
cd work/mix-stq/src/mixstq
cp ../../csrc/ggml-common.h .
python3 extract_iq2_tables.py
rm ggml-common.h
```

Expected 출력에 다음이 포함된다:
```
iq2s_grid: 1024 entries x 8 values, magnitudes [8, 25, 43]
iq3xxs_grid: 256 entries x 4 values, magnitudes [4, 12, 20, 28, 36, 44, 52, 62]
wrote tier_tables.json
```

이 크기 집합은 이미 파싱으로 검증된 값이다. 다르게 나오면 헤더 버전이 다른 것이므로 중단한다.

- [ ] **Step 3: Task 1의 테스트가 통과하는지 확인한다**

Run: `cd work/mix-stq && PYTHONPATH=src/mixstq:tests python3 tests/test_tier_encoders.py`
Expected: `PASS: three tiers registered, bpw exact, error monotone in bits`

오차가 단조롭지 않으면 Task 1의 `table.max()` 정규화가 그 티어에 맞지 않는 것이다. 그 경우 해당 티어의 `base` 스케일을 그리드 최대값으로 나누는 부분을 재확인한다.

- [ ] **Step 4: 러너에 테스트를 등록한다**

`scripts/run_tests.py`의 `OFFLINE` 리스트에 추가한다:

```python
OFFLINE = [
    "test_invariants.py",
    "test_imatrix_hook.py",
    "test_task_accuracy.py",
    "test_eval_tasks.py",
    "test_tier_encoders.py",
]
```

- [ ] **Step 5: 전체 오프라인 스위트와 린트를 돌린다**

Run: `cd work/mix-stq && python3 scripts/run_tests.py --offline && python3 -m ruff check .`
Expected: `5/5 passed` 그리고 `All checks passed!`

- [ ] **Step 6: 커밋**

```bash
cd work/mix-stq
git add src/mixstq/extract_iq2_tables.py src/mixstq/tier_tables.json scripts/run_tests.py
git commit -m "Extract the IQ2_S and IQ3_XXS grids from the ggml header"
```

---

### Task 3: 배분 플랜에 새 티어를 연결

**Files:**
- Modify: `src/mixstq/eval_mixed.py:161-172` (`apply_plan`의 티어 분기; `if tier == "ltc"`부터 `raise RuntimeError`까지)
- Modify: `src/mixstq/eval_tasks.py:135-148` (내부 플랜 정의 `def dense`부터 `plans = {...}`까지)
- Test: `tests/test_sweep_plans.py` (신규)

**Interfaces:**
- Consumes: `torch_iq2.quantize_rows(flat, local, tier=...)`, `torch_iq2.TIERS`
- Produces: `eval_tasks`의 플랜 이름 `dense | uniform_iq2 | mixed_ltc | iq2s_all | iq3_all | ltc_iq3`
- Produces: `apply_plan`이 인식하는 티어 문자열 `fp16 | iq2 | iq2_s | iq3_xxs | stq | ltc`

- [ ] **Step 1: 실패하는 테스트를 작성한다**

`tests/test_sweep_plans.py`:

```python
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

from eval_tasks import build_plans  # noqa: E402

failures = []

plans = build_plans(low_layers=6)
for name in ("dense", "uniform_iq2", "mixed_ltc", "iq2s_all", "iq3_all", "ltc_iq3"):
    if name not in plans:
        failures.append("plan %s missing" % name)

if "ltc_iq3" in plans:
    plan = plans["ltc_iq3"]
    if plan(0, "gate_up_proj") != "ltc":
        failures.append("ltc_iq3 must use ltc on low gate_up, got %s" % plan(0, "gate_up_proj"))
    if plan(9, "gate_up_proj") != "iq3_xxs":
        failures.append("ltc_iq3 must use iq3_xxs above the band, got %s" % plan(9, "gate_up_proj"))
    if plan(0, "down_proj") != "iq3_xxs":
        failures.append("ltc_iq3 must use iq3_xxs on down_proj, got %s" % plan(0, "down_proj"))

if "iq3_all" in plans and plans["iq3_all"](0, "gate_up_proj") != "iq3_xxs":
    failures.append("iq3_all must be uniform iq3_xxs")

if "dense" in plans and plans["dense"](0, "gate_up_proj") != "fp16":
    failures.append("dense must report fp16")

if failures:
    for line in failures:
        print("FAIL: " + line)
    raise SystemExit(1)
print("PASS: six sweep plans defined with the expected tier per layer band")
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cd work/mix-stq && PYTHONPATH=src/mixstq:tests python3 tests/test_sweep_plans.py`
Expected: FAIL — `ImportError: cannot import name 'build_plans'`

- [ ] **Step 3: `eval_tasks.py`에 `build_plans`를 추출한다**

`main()` 안의 플랜 정의 블록을 모듈 수준 함수로 올린다:

```python
def build_plans(low_layers):
    low = set(range(low_layers))

    def dense(_layer, _attribute):
        return "fp16"

    def uniform_iq2(_layer, _attribute):
        return "iq2"

    def mixed_stq(layer, attribute):
        if attribute == "down_proj":
            return "iq2"
        return "stq" if layer in low else "iq2"

    def mixed_ltc(layer, attribute):
        if attribute == "down_proj":
            return "iq2"
        return "ltc" if layer in low else "iq2"

    def iq2s_all(_layer, _attribute):
        return "iq2_s"

    def iq3_all(_layer, _attribute):
        return "iq3_xxs"

    def ltc_iq3(layer, attribute):
        if attribute == "down_proj":
            return "iq3_xxs"
        return "ltc" if layer in low else "iq3_xxs"

    return {
        "dense": dense,
        "uniform_iq2": uniform_iq2,
        "mixed_stq": mixed_stq,
        "mixed_ltc": mixed_ltc,
        "iq2s_all": iq2s_all,
        "iq3_all": iq3_all,
        "ltc_iq3": ltc_iq3,
    }
```

`main()`에서는 `plans = build_plans(args.low_layers)`로 대체하고, 기존 내부 정의는 삭제한다.

- [ ] **Step 4: `apply_plan`에 티어 분기를 추가한다**

`src/mixstq/eval_mixed.py`의 티어 분기를 교체한다:

```python
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
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `cd work/mix-stq && PYTHONPATH=src/mixstq:tests python3 tests/test_sweep_plans.py`
Expected: `PASS: six sweep plans defined with the expected tier per layer band`

- [ ] **Step 6: 러너 등록과 전체 스위트**

`scripts/run_tests.py`의 `OFFLINE`에 `"test_sweep_plans.py"`를 추가하고 실행한다.

Run: `cd work/mix-stq && python3 scripts/run_tests.py --offline && python3 -m ruff check .`
Expected: `6/6 passed`, `All checks passed!`

기존 `test_eval_tasks.py`도 여전히 통과해야 한다. `build_plans` 추출이 `score_item`이나 `render`를 건드리지 않았음을 이것으로 확인한다.

- [ ] **Step 7: 커밋**

```bash
cd work/mix-stq
git add src/mixstq/eval_tasks.py src/mixstq/eval_mixed.py tests/test_sweep_plans.py scripts/run_tests.py
git commit -m "Add the higher-bit tiers to the allocation plans"
```

---

### Task 4: Q1 스윕 실행 (GPU, ~$0.26)

**Files:**
- Modify: `src/mixstq/remote_tasks.sh` (팔 목록 기본값)
- Create: `artifacts/bit_frontier.json` (실행 산출물)

**Interfaces:**
- Consumes: Task 1–3의 전체 파이프라인, `artifacts/imatrix.json` + `imatrix.pt` (기존)
- Produces: `artifacts/bit_frontier.json` — `task_accuracy.py:compare` 형식, baseline은 `dense`

- [ ] **Step 1: GPU가 비어 있는 인스턴스를 확보한다**

```bash
cd work/mix-stq
python3 -m src.mixstq.vast_control search --gpu '' --max-price 0.30 --min-vram 40 --disk 80 --limit 5
python3 -m src.mixstq.vast_control create --offer <ID> --max-hourly 0.30 --min-vram 40 --disk 80 --confirm
python3 -m src.mixstq.vast_control list
```

`"status": "running"`이 되면 SSH로 다음을 확인한다:

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> \
  "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
```

Expected: `0 MiB, 0 %`. 다른 값이면 **즉시 destroy하고 다른 offer를 고른다.** v18에서 이 확인을 건너뛰어 $0.30을 잃었다.

- [ ] **Step 2: 파일을 올리고 스윕을 띄운다**

```bash
cd work/mix-stq
mkdir -p /tmp/mixstq_up/artifacts
cp src/mixstq/eval_tasks.py src/mixstq/eval_mixed.py src/mixstq/task_accuracy.py \
   src/mixstq/torch_iq2.py src/mixstq/torch_ltc.py src/mixstq/iq2xxs_tables.json \
   src/mixstq/tier_tables.json src/mixstq/remote_tasks.sh /tmp/mixstq_up/
cp artifacts/imatrix.json artifacts/imatrix.pt /tmp/mixstq_up/artifacts/
scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -P <PORT> -r /tmp/mixstq_up \
   root@<HOST>:/workspace/mixstq
```

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> \
 "cd /workspace/mixstq && MIXSTQ_REVISION=6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  MIXSTQ_MMLU=140 MIXSTQ_ARC=60 \
  MIXSTQ_ARMS=dense,mixed_ltc,iq2s_all,iq3_all \
  MIXSTQ_OUT=artifacts/bit_frontier.json \
  nohup bash remote_tasks.sh > /workspace/sweep.log 2>&1 & echo started"
```

`remote_tasks.sh`가 `MIXSTQ_OUT`을 읽도록 한 줄 수정한다:

```bash
OUT="${MIXSTQ_OUT:-artifacts/task_accuracy.json}"
```
그리고 `--out` 인자를 `"$OUT"`으로 바꾼다.

- [ ] **Step 3: 팔별 bpw 로그를 확인한다**

```bash
ssh ... "grep -E 'arm\]|bpw |accuracy ' /workspace/sweep.log"
```

각 양자화 팔마다 `bpw X.XXXX mean_error Y.YYYY` 줄이 **반드시** 나와야 한다.
없으면 양자화가 무효화된 것이고, v18에서 추가한 가드가 예외를 던졌을 것이다.
예상 bpw: `mixed_ltc` 1.8750, `iq2s_all` 2.5625, `iq3_all` 3.0625.

- [ ] **Step 4: 결과를 회수하고 인스턴스를 파괴한다**

```bash
cd work/mix-stq
scp -i ~/.ssh/id_ed25519 -P <PORT> root@<HOST>:/workspace/mixstq/artifacts/bit_frontier.json artifacts/
scp -i ~/.ssh/id_ed25519 -P <PORT> root@<HOST>:/workspace/sweep.log /tmp/sweep.log
python3 -m src.mixstq.vast_control destroy --id <INSTANCE_ID> --confirm
python3 -m src.mixstq.vast_control list
```

Expected: `list`가 `[]`와 `my total burn so far: $0.0000`을 출력한다.

- [ ] **Step 5: Q1 게이트를 판정한다**

```bash
cd work/mix-stq
python3 - <<'PY'
import json, sys
sys.path.insert(0, "src/mixstq")
from task_accuracy import compare
d = json.load(open("artifacts/bit_frontier.json"))
dense = d["arms"]["dense"]["accuracy"]
for name, stats in d["arms"].items():
    print("%-12s %.4f  delta_vs_dense %+.4f" % (name, stats["accuracy"], stats["accuracy"] - dense))
PY
```

손실이 5%p 이내인 팔이 하나라도 있으면 Q1 통과, 그 팔의 bpw가 사용 가능 지점이다.
전부 5%p를 넘으면 **Task 5를 건너뛰고 Task 6으로 가서 종료를 문서화한다.**

- [ ] **Step 6: 커밋**

```bash
cd work/mix-stq
git add artifacts/bit_frontier.json src/mixstq/remote_tasks.sh
git commit -m "Measure the task-accuracy loss curve across bit budgets"
```

---

### Task 5: Q2 확인 — 사용 가능 지점에서만, 600문항

**전제조건:** Task 4의 Q1 게이트 통과. 실패했다면 이 태스크를 실행하지 않고 Task 6으로 간다.

**Files:**
- Create: `artifacts/ltc_at_usable_bits.json` (실행 산출물)

**Interfaces:**
- Consumes: Task 4가 식별한 사용 가능 bpw
- Produces: `artifacts/ltc_at_usable_bits.json` — baseline이 고정 코드북 팔인 `compare` 출력

- [ ] **Step 1: 문항 수를 600으로 올려 3팔을 실행한다**

사용 가능 지점이 IQ3_XXS(3.0625 bpw)로 나온 경우:

```bash
ssh ... "cd /workspace/mixstq && MIXSTQ_REVISION=6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  MIXSTQ_MMLU=420 MIXSTQ_ARC=180 \
  MIXSTQ_ARMS=dense,iq3_all,ltc_iq3 \
  MIXSTQ_OUT=artifacts/ltc_at_usable_bits.json \
  nohup bash remote_tasks.sh > /workspace/q2.log 2>&1 & echo started"
```

600문항은 5%p 차이를 80% 검정력으로 검출한다(spec §6의 검증된 계산: 필요 불일치 쌍 124개, 필요 문항 616개). 3%p를 목표로 하지 않는 이유는 1,733문항이 필요하고 5%p 미만의 코드북 차이는 배포 결정을 바꾸지 않기 때문이다.

- [ ] **Step 2: 결과를 회수하고 즉시 파괴한다**

Task 4 Step 4와 동일한 절차. `list`가 `[]`임을 확인한다.

- [ ] **Step 3: Q2 게이트를 판정한다**

```bash
cd work/mix-stq
python3 - <<'PY'
import json, sys
sys.path.insert(0, "src/mixstq")
d = json.load(open("artifacts/ltc_at_usable_bits.json"))
for label, cmp in d["comparisons"].items():
    print("%s delta=%+.4f CI[%+.4f,%+.4f] p=%.4f significant=%s" % (
        label, cmp["accuracy_delta"], cmp["ci_95"][0], cmp["ci_95"][1],
        cmp["mcnemar_p"], cmp["significant"]))
PY
```

`significant`가 True면 LTC 우위 확립. False면 동등이며, 그것이 결론이다.
McNemar와 CI 중 하나만 만족하는 경우는 미확정으로 기록하고 표본 확대를 다음 작업으로 남긴다.

- [ ] **Step 4: 커밋**

```bash
cd work/mix-stq
git add artifacts/ltc_at_usable_bits.json
git commit -m "Compare the learned codebook at the usable bit budget"
```

---

### Task 6: v19 연구 기록 작성과 푸시

**Files:**
- Create: `docs/mix-stq-v19-bit-frontier.md`

**Interfaces:**
- Consumes: `artifacts/bit_frontier.json`, 있으면 `artifacts/ltc_at_usable_bits.json`

- [ ] **Step 1: 저장소 문서 형식을 그대로 따라 기록을 쓴다**

기존 v18 문서와 동일한 구조를 쓴다. 필수 절:

1. 결론 (한 문단, 수치 포함)
2. 측정 조건 표 (모델, 리비전, 문항 수, GPU, 통계 방법)
3. 손실 곡선 표 — 팔별 bpw, mean_error, 정확도, dense 대비 delta
4. Q1 게이트 판정, Q2 게이트 판정 (실행한 경우)
5. **검증 로그** 표 — 주장 / 근거 / 상태
6. **측정하지 않은 것** 절 — 명시적으로
7. 비용 표 — 인스턴스별 시간과 비용, 총계
8. 다음에 할 일

v18에서 확립한 규칙을 유지한다: 측정하지 않은 것을 측정한 것처럼 쓰지 않고,
외부 웹 출처 수치는 "측정값 아님"으로 명시한다.

- [ ] **Step 2: Q1이 실패한 경우의 기록**

전 팔이 5%p를 넘었다면 결론은 다음이다: **이 배분 패밀리는 2.0–3.1 bpw 구간
전체에서 배포 불가이며, LTC 코드북 연구는 태스크 정확도 축에서 종료한다.**
이것을 완곡하게 쓰지 않는다. spec §5가 이 판정을 사전에 확정했다.

- [ ] **Step 3: 검증을 돌린다**

Run: `cd work/mix-stq && python3 scripts/run_tests.py --offline && python3 -m ruff check .`
Expected: 전체 통과

- [ ] **Step 4: 커밋하고 푸시한다**

```bash
cd work/mix-stq
git add docs/mix-stq-v19-bit-frontier.md docs/superpowers/
git commit -m "Record the bit-budget frontier result"
git push origin main
git fetch origin -q && git log --oneline origin/main -1
```

Expected: `origin/main`이 새 커밋을 가리킨다.

---

## Self-Review

**1. Spec 커버리지**

| Spec 절 | 담당 태스크 |
|---|---|
| §3 실제 ggml 형식 사용 | Task 2 (헤더에서 추출, 크기 집합 검증) |
| §4 6개 팔 스윕 설계 | Task 3 (`build_plans`), Task 4 (4팔 실행) |
| §5 Q1/Q2 게이트 사전 확정 | Task 4 Step 5, Task 5 Step 3 |
| §6 검증된 표본 수 | Task 5 Step 1 (600문항, 근거 명시) |
| §7 비용 모델 | Task 4, Task 5 (각 ~$0.26) |
| §8 하지 않는 것 | 계획에 해당 태스크 없음 (의도적) |
| §9 리스크 완화 | Task 1–2 (bpw/단조성 테스트), Task 4 Step 1 (GPU 점유 확인), Step 3 (bpw 로그), Step 4 (파괴 확인) |

**2. 플레이스홀더 스캔**

`<ID>`, `<PORT>`, `<HOST>`, `<INSTANCE_ID>`는 런타임에 결정되는 값이고
얻는 명령이 바로 위에 있으므로 플레이스홀더가 아니다. TBD/TODO/"적절히 처리"는 없다.
모든 코드 스텝에 실제 코드가 들어 있다.

**3. 타입 일관성**

- `quantize_rows(matrix, importance, tier=...)` — Task 1에서 정의, Task 3에서 동일 서명 사용
- `TIERS[tier]["bpw"]` / `["lane"]` / `["table"]` — Task 1 정의, Task 2·3에서 동일 키
- `build_plans(low_layers)` → `dict[str, callable]`, 콜러블은 `(layer, attribute) -> str` — Task 3 정의, 테스트와 `main()`에서 동일
- 티어 문자열 `"iq2_s"`, `"iq3_xxs"`는 `TIERS` 키와 `build_plans` 반환값에서 철자가 일치한다
- `tier_tables.json` 키 `"iq2s_grid"`, `"iq3xxs_grid"`는 `TIERS[*]["table"]` 값과 일치한다

**4. 알려진 환경 제약**

- `exec_command`는 약 10초에서 끊긴다. GPU 실행은 `nohup` + 로그 폴링으로 한다.
- LSP(basedpyright, bash-language-server)가 없어 `apply_patch`가 매번 ERR 메시지를 낸다. **패치는 대개 성공한다.** 재시도하지 말고 읽기나 ruff로 확인한다.
- macOS에 `setsid`가 없다.
- zsh에서 글롭은 인용한다.
- 로컬 torch 2.2.2 + NumPy 2.5.2 경고는 무해한 기존 상태다. transformers 5.16.1은 torch >= 2.5를 요구하므로 HF 의존 테스트는 로컬에서 돌지 않는다.
