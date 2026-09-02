# MIX-STQ v0.27 사전등록: Qwen3.8-27B 실제 GGUF 3/4/5 bpw 프론티어

**사전등록일** 2026-08-31
**상태** 실제 GGUF 결과 미열람, 유료 실행 전 고정
**주 질문** 같은 BF16 원본과 같은 llama.cpp importance matrix에서 만든
`IQ3_XXS`, `IQ4_XS`, `Q4_K_M`, `Q5_K_M`이 BF16 GGUF 대비 정확도·perplexity·속도에서
어떤 물리적 비트/품질 프론티어를 이루는가?

v0.26은 Qwen3.8-27B의 MLP 192개를 Python에서 참조 IQ3_XXS 상태 공간으로
재구성한 800문항 결과다. 이 문서는 그 결과를 packed GGUF나 llama.cpp 실행 결과로
소급 해석하지 않는다. v0.27은 표준 llama.cpp converter와 quantizer가 실제로 만든
파일만 평가하며, 4–5 bpw 대조군까지 같은 실행 경로로 측정한다.

## 1단계: 소스·모델·비교 팔을 고정

| 항목 | 고정값 |
|---|---|
| 원본 모델 | `Qwen/Qwen3.8-27B` |
| 모델 revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| text architecture | `Qwen3_5ForConditionalGeneration`의 `QWEN35` text model |
| llama.cpp commit | `580e88d8b7dece7099d9b62323521d0254ff3615` |
| 변환 기준 | `convert_hf_to_gguf.py --outtype bf16 --no-mtp` |
| 기준 팔 | text-only `BF16` GGUF |
| 양자화 팔 | `IQ3_XXS`, `IQ4_XS`, `Q4_K_M`, `Q5_K_M` |
| 양자화 출발점 | 모든 팔이 같은 BF16 GGUF에서 직접 생성 |
| MTP | text 비교에서 제외; converter의 `--no-mtp`로 고정 |
| vision projector | 한 번 별도 생성·보존하며 text 파일 bpw 분모와 벤치마크에서 제외 |

Qwen3.8-27B의 64개 language layer, hidden size 5,120, intermediate size 17,408,
24 attention heads, 4 KV heads, head dimension 256, vocabulary 248,320이라는 구조는
고정 revision의 config와 생성된 GGUF metadata에서 모두 확인한다. 3개 linear-attention
layer 뒤 1개 full-attention layer가 반복되는 구조와 untied output head도 tensor inventory에
기록한다.

양자화 파일을 다시 양자화하는 requantization은 금지한다. arm마다 quantizer가 읽은 input
SHA-256이 동일한 BF16 GGUF SHA-256과 일치하지 않으면 해당 arm을 무효로 한다. quantizer
type 이름은 위 네 문자열과 정확히 일치해야 하며, 유사 bpw의 다른 포맷으로 대체하지 않는다.

## 2단계: calibration·호스트·실행 환경을 고정

importance matrix는 v0.26의 PyTorch `.pt`를 변환하지 않고, 고정한 llama.cpp의
`llama-imatrix`가 직접 만든 GGUF 형식을 사용한다. calibration corpus는
`src/mixstq/llama_calibration.py`가 다음 세 소스에서 source order상 첫 유효 텍스트를
각 32개씩 선택해 만든다.

| domain | dataset/config/split | revision | field | 규칙 |
|---|---|---|---|---|
| wiki | `Salesforce/wikitext`, `wikitext-2-raw-v1`, `train` | `b08601e04326c79dfdd32d625aee71d232d685c3` | `text` | 정규화 후 Unicode 200자 이상, 32개 |
| code | `codeparrot/codeparrot-clean-valid`, `train` | `4db92d2ec0c1b4c41eeb439cfae16854511d9dcd` | `content` | 정규화 후 Unicode 200자 이상, 32개 |
| chat | `HuggingFaceH4/ultrachat_200k`, `train_sft` | `8049631c405ae6576f93f445c6b8166f76f5505a` | `prompt` | 정규화 후 Unicode 200자 이상, 32개 |

총 96개 record의 원문 SHA-256, 순서, domain, corpus SHA-256과 aggregate SHA-256을
manifest에 기록한다. record separator는 builder가 충돌 여부를 검사해 고른 16 byte 이하의
control separator다. MMLU나 ARC 문항은 calibration에 넣지 않는다. corpus 또는 manifest가
이미 있으면 덮어쓰지 않고 실패하며, 두 파일 중 하나만 출판되는 상태도 실패다.

`llama-imatrix` 고정값은 context 512, batch 512, ubatch 128, 정확히 128 chunks, threads 32,
`--no-ppl`, GGUF output이다. 80 GB 이상 GPU에서는 전 layer offload를 요청하되, 실제 offload
layer 수와 peak VRAM을 로그에서 회수한다. 생성된 단일 imatrix SHA-256을 네 양자화 팔이
공유한다.

1차 Vast.ai 호스트 기준은 다음과 같다.

| 자원 | 하한/상한 |
|---|---:|
| GPU | 단일 NVIDIA GPU, VRAM 80 GB 이상 |
| system RAM | 96 decimal GB 이상 |
| CPU | Vast `cpu_cores` 16 이상, 가능하면 effective 32 이상 |
| disk | 300 GB 이상 |
| download | 500 Mbps 이상 |
| reliability | 0.98 이상 |
| compute 가격 | $1.20/hour 이하 |

offer는 생성 직전에 같은 조건으로 재검증한다. `nvidia-smi`에 다른 compute process가 있거나
GPU·RAM·disk가 하한보다 작으면 모델 다운로드 전에 인스턴스를 폐기한다. 80 GB offer가
없다는 이유로 24 GB GPU를 조용히 대신 쓰지 않는다. 24–48 GB 변환 전용 실행은 별도
run label과 별도 사전등록 없이는 이번 BF16 비교에 합치지 않는다.

## 3단계: 변환·파일 무결성·배포 smoke gate를 고정

고정 llama.cpp commit을 detached checkout하고 CUDA release binary를 빌드한다. 실행 전에
converter, `llama-imatrix`, `llama-quantize`, `llama-cli`, `llama-server`,
`llama-perplexity`, `llama-bench`, `llama-gguf-split`의 존재와 `--help`를 확인한다.
모델 snapshot도 고정 revision으로 내려받고 snapshot identity를 기록한다.

순서는 다음과 같다.

1. text-only BF16 GGUF를 `--no-mtp`로 한 번 만든다.
2. vision projector를 별도 파일로 한 번 만든다.
3. BF16 text GGUF와 고정 corpus로 imatrix를 한 번 만든다.
4. 같은 BF16 input과 같은 imatrix에서 네 양자화 arm을 각각 직접 만든다.
5. 모든 파일의 SHA-256, byte size, GGUF version, architecture, quantization type,
   tensor 수와 tensor element 합계를 수집한다.

각 text GGUF에 두 bpw를 병기한다.

- **physical file bpw** = `전체 파일 bytes * 8 / GGUF tensor element 합계`
- **tensor payload bpw** = `tensor data 영역 bytes * 8 / GGUF tensor element 합계`

projector는 두 계산에서 제외한다. marketing상의 이름이나 이론 bpw를 실측 file bpw로
대체하지 않는다. BF16 input SHA, imatrix SHA, quantizer stdout/stderr, wall time, peak RSS,
peak VRAM과 최종 SHA를 arm별 manifest에 묶는다.

모든 arm은 benchmark 전에 독립 `llama-cli` process로 다음 smoke를 통과해야 한다.

- GGUF가 `QWEN35`로 load되고 예상 tensor가 모두 읽힘
- seed 22, temperature 0의 짧은 text completion이 비어 있지 않음
- stderr에 tensor shape 오류, unsupported architecture, NaN/Inf 또는 abort가 없음
- 종료 code 0과 실제 prompt/eval timing이 기록됨

Python 참조 인코더와 llama.cpp packed byte의 bit-for-bit parity는 별도 검증이 없는 한
주장하지 않는다. 이번 배포 팔의 권위 있는 형식은 고정 llama.cpp quantizer가 생성하고
고정 llama.cpp runtime이 실제 load한 GGUF다. smoke를 실패한 파일은 정확도나 속도 표에
넣지 않는다.

## 4단계: 동일 평가와 판정 규칙을 고정

### 4.1 800문항 Top-1

v0.26과 같은 MMLU 570 + ARC-Challenge 230, 총 800문항과 ordered fingerprint
`a72515282c6fc20f34188b3102d99468ab2b02266105ed9c6e4ec405fbad8fd0`을 사용한다.
prompt serialization과 item order도 동일하게 유지한다.

각 GGUF는 새 `llama-server` process에서 평가한다. `/tokenize`로 `A`, `B`, `C`, `D`가
서로 다른 단일 token인지 먼저 확인한다. `/completion`은 `n_predict=1`, greedy sampling,
seed 22, `cache_prompt=false`, repetition penalty 1.0을 쓰고 네 token에 같은 `+100` bias를
준다. 응답의 generated token과 top-4 token id가 후보 집합 안인지 검사한다. 이 bias는
후보 외 token을 밀어내되 후보 네 개 사이의 원래 logit 순서는 보존한다. 조건을 만족하지
않는 응답은 오답으로 바꾸지 않고 protocol failure로 중단한다.

arm별 correctness vector를 보존한다. BF16과 각 quant arm의 paired 차이
`delta = accuracy(BF16) - accuracy(quant)`에 대해 seed 22, 10,000회 paired percentile
bootstrap 95% CI와 exact two-sided McNemar를 계산한다. 주 판정은 다음과 같다.

| 조건 | 판정 |
|---|---|
| CI upper `<= +0.0200` | BF16 대비 2%p 비열등성 통과 |
| CI lower `> 0`이고 McNemar `p < 0.05` | 유의한 성능 손실 |
| 그 외 | 미확정 |

### 4.2 held-out perplexity와 처리량

perplexity는 calibration과 겹치지 않는 pinned WikiText-2 `test` split을 source order로
정규화한 별도 corpus에서 측정한다. 같은 `llama-perplexity` binary, context/chunk 설정과
같은 GPU offload를 쓰며 absolute PPL과 BF16 대비 상대 증가율을 기록한다.

`llama-bench`는 각 arm에 prompt 512 tokens, generation 128 tokens, 5 repetitions로
실행한다. prompt processing tok/s와 generation tok/s의 raw repetitions 및 median,
runtime peak VRAM, host RAM을 보존한다. 실행 사이 server/process가 남아 있으면 다음 arm을
시작하지 않는다.

Top-1, PPL, 처리량 중 하나라도 누락된 arm은 “전체 비교 완료”로 표시하지 않는다.
Flint Chart에는 실측된 arm만 같은 축에 표시하고, MLP-only v0.26 점과 실제 GGUF 점을
같은 series로 오인하게 합치지 않는다.

## 5단계: 보존·Terminal Bench·중단 규칙을 고정

네 quantized GGUF는 `llama-gguf-split`로 shard당 8 GiB 이하로 분할한다. 한 arm씩
trusted local machine으로 전송하고, local write token으로
`topabaem/mix-stq-artifacts/paid-run/qwen38-gguf-frontier-v27/` 아래에 업로드한다.
Hugging Face write token은 Vast 호스트의 파일, 환경, 명령 인자 또는 로그에 두지 않는다.

업로드 후 public unauthenticated re-download로 모든 shard SHA-256과 aggregate manifest를
다시 확인한다. 로컬 여유 공간이 작으므로 shard를 순차 처리할 수 있지만, local shard를
지우기 전 해당 shard의 public re-download hash가 일치해야 한다. Vast의 원본 monolithic
GGUF는 전체 arm의 HF 재검증이 끝날 때까지 유지한다. BF16 GGUF는 원본 model revision과
재생성 manifest를 보존하고, 네 quantized arm과 calibration/imatrix/로그/결과를 우선
영구 보존한다.

`topabaem@100.73.38.99`는 16 GB V100과 약 29 GiB RAM이므로 Qwen3.8-27B BF16 server를
수용하는 모델 호스트로 사용하지 않는다. 모델 server는 검증된 Vast 호스트에 두고, 대상
서버는 Harbor/Docker/Claude Code harness만 실행한다. Claude Code의 Anthropic
`/v1/messages` 계약과 llama.cpp의 OpenAI-compatible API는 직접 호환되지 않으므로,
계약 smoke를 통과한 실험용 gateway를 사이에 둔다. 이 경로의 결과는 공식 Claude 모델
leaderboard parity로 주장하지 않는다.

Terminal Bench 입력은 Harbor v0.22.0 commit
`4407eb5227a2ff4f0d3f16b2eb48849382fdf276`, 89-task dataset digest
`sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`,
dataset snapshot `320a8be8b625ee8eb46481f7a397648d7d085775`로 고정한다. 모든 container
image가 mutable tag이므로 실행 전에 실제 OCI digest를 resolve·기록해야 한다. 선택지 강제,
답안 regex 보정, BF16 답 복사, task별 수동 retry는 금지한다. 모든 arm은 같은 task set,
timeout, agent scaffold, prompt policy와 concurrency를 사용하고 task별 reward, pass/fail,
wall time, timeout/error를 보존한다.

부분 실행은 `partial`로만 보고한다. Terminal Bench full 중 한 arm이 실패하면 원인을
기록하고 같은 고정 조건으로 해당 arm 전체를 다시 실행하며, 성공한 task만 골라 합치는
resume 결과를 최종 full로 승격하지 않는다.

최종 보존 manifest와 Terminal Bench 결과를 회수한 뒤 Vast.ai 인스턴스를 destroy하고
활성 instance 목록 `[]`와 burn `$0.0000`을 확인한다. 시간과 비용은 API 및 로그에서
관찰된 값만 실측으로 표기한다. 실행 전 예상치는 다음 범위로만 사용하며 결과로 인용하지
않는다.

- GGUF 변환·imatrix·4-arm quantization: 약 4–10시간
- 800문항·PPL·llama-bench: 약 3–12시간
- Terminal Bench 2.1 full: task 설정상 arm당 agent timeout 합계 42.21시간,
  verifier timeout 합계 41.43시간이며 5개 arm 순차 실행의 설정상 최악 상한은 약 418시간이다.
  smoke와 소규모 pilot으로 실측 속도·비용을 얻고 사용자 비용 승인을 받은 뒤 full을 시작한다.

실제 측정이 이 범위를 벗어나도 표본·arm·판정 규칙을 바꾸지 않는다.

---

## 개정 1 (2026-09-02): top-4 확률의 의미 정정

4.1의 top-4 검사를 개정한다. 고정 commit `580e88d8b7dece7099d9b62323521d0254ff3615`에서
`post_sampling_probs`를 보내지 않는 요청의 `top_logprobs`는 sampling 이전 raw logit에서
계산되어 `logit_bias`를 반영하지 않는다는 것을 소스와 실측으로 확인했다. 따라서 생성
token이 후보 집합 안인지 검사하는 규칙은 유지하고, top-4 token id 집합이 후보 집합과
같은지 검사하는 규칙은 제거해 그 목록을 pre-sampling 진단값으로 기록한다. 요청 payload,
표본 800문항, fingerprint, arm 집합, 판정 규칙은 바꾸지 않는다.

이 개정은 유료 실행 전, v0.27 GGUF 결과를 하나도 관찰하지 않은 시점에 이루어졌다. 근거,
소스 인용, probe 응답과 대안 기각 사유는
[`mix-stq-v27-amendment-1-top4-semantics.md`](mix-stq-v27-amendment-1-top4-semantics.md)에 있다.

---

## 개정 2 (2026-09-02): imatrix chunk 수의 의미와 PPL corpus 직렬화

2단계의 "정확히 128 chunks"를 개정한다. 고정값은 argv `--chunks 128` 그대로이며 이 값은
상한이다. 실현 chunk 수는 `floor(token_count / 512)`로 결정되고 반드시 기록한다. 1차 유료
실행에서 고정 corpus(sha256 `79f0c5cf125b9da642e82519e8630885c67c75336dd628eba69a898cdac681d5`,
46,981 byte)는 10,523 token으로 측정되어 실현 20 chunk였고, argv는 한 글자도 바뀌지 않았다.
저장소 gate `exact_tokenizer_preflight`는 이미 `0 < token_count <= 128 * 512`를 요구한다.

4.2의 held-out PPL corpus 직렬화를 1차 실행에서 실제로 쓴 규칙으로 고정한다. WikiText-2
`test` split, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, 4,358 record를 source
order로 구분자 없이 연결한 뒤 `mixstq.llama_calibration.normalize_text`를 전체에 한 번
적용한다. 결과 byte 열의 sha256은
`03492eaf99762251b0c9ed3bc4229294e7f3a03c5ec8cb9cdb61f54999539e11`이다. 소스, split, 순서,
정규화 여부, 측정 설정은 바뀌지 않는다.

이 개정은 Task 5 파일 생성과 BF16 Top-1 696/800 관찰 이후, 그러나 어떤 양자화 arm의 Top-1,
어떤 arm의 PPL, 어떤 arm의 llama-bench 값도 관찰하기 전에 이루어졌다. 따라서 어떤 비교
결과도 이 결정에 영향을 줄 수 없었다. 근거와 증거는
[`mix-stq-v27-amendment-2-imatrix-chunks-and-ppl-serialization.md`](mix-stq-v27-amendment-2-imatrix-chunks-and-ppl-serialization.md)에 있다.
