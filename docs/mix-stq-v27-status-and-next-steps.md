# MIX-STQ v0.27 연구 상태와 다음 작업

**기준일:** 2026-08-31
**저장소:** `Topabaem05/mix-stq`
**현재 단계:** 유료 실행 전 코드 기반 완료, 실제 GGUF 측정 미시작

이 문서는 다음 세션이 대리 지표를 실제 배포 결과로 오인하지 않고 연구를 이어가기 위한
공개 handoff다. MIX-STQ 양자화 연구는 `mix-stq`에서 계속하며, Pacific 저장소와 분리한다.

## 1단계: 지금까지 확인된 연구 결과

Qwen3.8-27B v0.26의 권위 있는 측정은 BF16 모델에서 MLP 192개 텐서만 참조 IQ3_XXS
상태 공간으로 재구성한 800문항 결과다.

| arm | 범위 | Top-1 | BF16 대비 |
|---|---|---:|---:|
| dense BF16 | 원본 BF16 | 696/800 (87.000%) | 기준 |
| reference IQ3_XXS | MLP-only 재구성 | 693/800 (86.625%) | -0.375%p |

평가 fingerprint는
`a72515282c6fc20f34188b3102d99468ab2b02266105ed9c6e4ec405fbad8fd0`이다.
이 결과는 packed GGUF, llama.cpp 실행, 자유형 생성 또는 Terminal Bench 결과가 아니다.
현재 실제 v0.27 BF16/IQ3_XXS/IQ4_XS/Q4_K_M/Q5_K_M GGUF 점수는 모두 **미측정**이다.

## 2단계: 완료된 유료 실행 전 기반

고정 계약은 다음과 같다.

- 모델: `Qwen/Qwen3.8-27B`
- 모델 revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- llama.cpp commit: `580e88d8b7dece7099d9b62323521d0254ff3615`
- text 변환: BF16, `--no-mtp`
- 직접 양자화 arm: `IQ3_XXS`, `IQ4_XS`, `Q4_K_M`, `Q5_K_M`
- 모든 quant arm은 같은 BF16 GGUF와 같은 imatrix에서 직접 생성하며 requantization을 금지

완료된 로컬 커밋:

| commit | 내용 | 관찰된 검증 |
|---|---|---|
| `17c38d6` | Vast host 자원 필터 | 후속 보안 테스트에 포함 |
| `e76349f` | 결정적 calibration builder | 후속 publication 테스트에 포함 |
| `4b8c79e` | calibration 원자 출판/commit marker | 후속 focused 테스트에 포함 |
| `0c249bf` | domain round-robin, 512-byte cap, hash/용량 계약 | 22 focused, offline 11/11 |
| `0fa807a` | Vast secret/state/수치 검증 강화 | 91 focused, offline 11/11, read-only API 정상 |
| `b430a65` | pinned GGUF run planner와 model/host/token preflight | 69 focused; worker evidence상 offline 11/11/Ruff 통과 |

Calibration은 wiki/code/chat 각 32개를 source order로 고른 뒤
`wiki[i] -> code[i] -> chat[i]`로 직렬화한다. record는 UTF-8 512 bytes 이하이며 corpus는
65,536 bytes를 넘으면 실패한다. 이것은 정확 토큰 수를 대신하지 않으며, 실행 전 pinned
`llama-tokenize --show-count`로 65,536 tokens 이하인지 별도 확인한다.

GGUF planner는 `bootstrap`, `calibration`, `convert`, `imatrix`, `quantize`, `smoke`,
`audit`, `split`의 여덟 phase를 credential 없는 argv로 만든다. Vast 기본 상태 파일은
XDG state 경로에 private mode로 원자 저장하며 현재 활성 Vast instance는 없다.

## 3단계: 다음에 구현할 코드

다음 순서를 바꾸지 않는다.

1. `gguf_audit.py`: file SHA, architecture, tensor type/element, physical file bpw와 payload bpw를
   streaming 방식으로 검증한다.
2. `eval_llama_server.py`: 고정 800문항 fingerprint, 자유형 llama-server protocol,
   correctness vector, 원자 resume/completion marker를 구현한다.
3. paired statistics: BF16 대비 delta, seed 22의 10,000회 paired bootstrap 95% CI,
   exact two-sided McNemar를 산출한다.
4. held-out PPL/throughput: pinned WikiText-2 test corpus, `llama-perplexity`, prompt 512,
   generation 128, 5회 `llama-bench` raw 결과를 보존한다.
5. artifact preservation: 8 GiB 이하 shard, local hash, public unauthenticated HF 재다운로드
   hash가 모두 닫힌 뒤에만 원격 원본과 Vast instance를 제거한다.

500편 이상 논문을 체계적으로 수집·독해했다는 목표는 아직 완료되지 않았다. 기존 문헌
검토를 실제 500편 corpus로 과장하지 말고, DOI/arXiv ID, 연도, 방법, bit-width, 모델,
평가셋, 핵심 결과와 재현성 상태를 가진 중복 제거 bibliography를 별도 구축해야 한다.

## 4단계: 실제 Vast GGUF 실험

유료 실행 전에는 candidate HEAD 전체 focused/offline/Ruff/CLI 검사, 독립 review, clean
worktree와 원격 SHA 일치를 먼저 확인한다. Vast offer는 단일 NVIDIA GPU 80 GB 이상,
system RAM 96 decimal GB 이상, CPU 16 cores 이상, disk 300 GB 이상, download 500 Mbps
이상, reliability 0.98 이상, 가격 $1.20/hour 이하를 모두 만족해야 한다.

실행 순서는 BF16 변환, 정확 tokenizer preflight, 단일 imatrix, 네 direct quant arm,
arm별 독립 smoke/audit, 800 Top-1, PPL, throughput, HF public hash 검증이다. 하나라도 누락된
arm은 완료된 frontier로 보고하지 않는다. v0.26 MLP-only 결과와 실제 GGUF 결과는 같은
series로 합치지 않는다.

## 5단계: Terminal Bench와 최종 보고

Terminal Bench 2.1은 89 tasks를 자유형으로 실행한다. target 서버의 16 GB V100은 모델
호스트가 아니라 Harbor/Docker/Claude Code harness 호스트로만 사용하고, Vast의
llama-server와 Anthropic-compatible experimental gateway로 연결한다. 먼저 contract smoke와
소규모 pilot을 실행하고, 관찰된 task당 시간·오류율·비용으로 full 5-arm 예산을 다시 계산한
뒤 사용자 승인을 받는다. 설정상 순차 최악 상한은 약 418시간이므로 기존 12–36시간 추정은
사용하지 않는다.

고정 identity는 Harbor commit `4407eb5227a2ff4f0d3f16b2eb48849382fdf276`, dataset
snapshot `320a8be8b625ee8eb46481f7a397648d7d085775`, sorted task fingerprint
`8135dcdf6a6a32585be20798426d9f258a34fa6d5672318fd6530057962ad8de`, image-reference
fingerprint `7120da32a5ffe25c9c6023a921803c64877f1c6612107e5236d53065e4c6d6fc`, Claude Code
`2.1.123`이다. 89개 image reference는 모두 mutable tag이므로 pre-pull 후 실제 OCI digest를
manifest에 고정한다.

최종 결과물은 다섯 arm의 실제 physical/payload bpw, Top-1, BF16 delta/CI/McNemar, PPL,
처리량, VRAM/RAM, wall time, 비용, public artifact URL을 포함한다. Flint Chart에는 실제 GGUF
Top-1만 한 chart에 표시하고 v0.26 MLP-only 값은 별도 주석/series로 분리한다. 마지막 단계는
Vast destroy와 활성 instance 0 확인이다.

## 다음 세션 시작 체크리스트

- 원격 `main` SHA와 local HEAD 일치 확인
- worktree clean 및 내부 orchestration artifact 미추적 확인
- Task 3B에 해당하는 GGUF auditor와 800-item evaluator를 서로 분리된 커밋으로 구현
- 실제 유료 실행 전 전체 독립 review와 run-lock 생성
- credential은 local trusted host 밖으로 전달하거나 저장하지 않기
- 측정하지 않은 값은 `미측정`, 예상치는 `예상`, 실제 로그값만 `실측`으로 표기
