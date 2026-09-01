# MIX-STQ v0.27 사전등록 개정 1: top-4 확률은 pre-sampling 진단값이다

**개정일** 2026-09-02
**대상 문서** [`mix-stq-v27-gguf-frontier-preregistered.md`](mix-stq-v27-gguf-frontier-preregistered.md) 4.1
**상태** 유료 실행 전, v0.27 GGUF 결과를 하나도 관찰하지 않은 시점의 개정
**한 줄 변경** 생성 token의 후보 집합 포함 검사는 유지하고, 응답 top-4 token id와
후보 집합의 동일성 검사는 제거해 해당 필드를 pre-sampling 진단값으로 기록한다.

이 개정은 요청 payload를 바꾸지 않는다. `n_predict=1`, `temperature=-1.0`, `seed=22`,
`cache_prompt=false`, repetition/presence/frequency penalty, 네 후보 token에 대한 `+100.0`
logit bias, `n_probs=4`, `return_tokens=true`는 byte 단위로 동일하다. 표본 800문항,
ordered fingerprint, arm 집합, 2%p 비열등 판정 규칙, paired bootstrap과 McNemar 설정도
바꾸지 않는다.

## 1단계: 사전등록이 가정한 것

4.1은 "응답의 generated token과 top-4 token id가 후보 집합 안인지 검사한다"고 고정했다.
구현은 이를 가장 엄격하게 읽어 `sorted(top_ids) == sorted(letter_ids)`를 protocol gate로
넣었다. 이 가정은 `logit_bias`가 응답의 `top_logprobs` 목록에도 반영되어, `+100` bias를
받은 네 후보가 항상 상위 4개를 차지한다는 것이었다.

## 2단계: 고정 commit에서 관찰한 사실

llama.cpp를 고정 commit `580e88d8b7dece7099d9b62323521d0254ff3615`(commit date
2026-08-31 12:17:51 +0200)로 detached checkout해 로컬 빌드하고, 평가기와 동일한 요청
흐름을 재현했다. 소스와 실측이 같은 결론을 준다.

### 2.1 소스 경로

| 위치 | 내용 |
|---|---|
| `tools/server/server-context.cpp:1779` | `need_pre_sample_logits = n_probs > 0 && !post_sampling_probs` |
| `tools/server/server-context.cpp:1784` | 위 조건이 참이면 backend sampling을 끈다 |
| `tools/server/server-context.cpp:1985` | 해당 분기에서 `get_token_probabilities(ctx_tgt, idx, n_probs_request)` 호출 |
| `tools/server/server-common.cpp:1501-1504` | `get_token_probabilities`가 `llama_get_logits_ith`의 raw logits를 읽는다 |
| `common/sampling.cpp:130-155` | `set_logits`가 sampler chain이 다룰 candidate 배열 `cur`를 따로 만든다 |
| `src/llama-sampler.cpp:3902` | `llama_sampler_logit_bias_apply`는 그 `llama_token_data_array * cur_p` 사본만 수정한다 |
| `tools/server/server-task.cpp:274, 296` | `post_sampling_probs`에 따라 `logprob`/`prob`, `top_logprobs`/`top_probs` 키가 결정된다 |

즉 `post_sampling_probs`를 보내지 않는 우리 payload에서 `top_logprobs`는 **sampling 이전의
raw logit**에서 계산되며 `logit_bias`가 반영되지 않는다. bias는 sampler chain이 들고 있는
candidate 배열 사본에만 적용되므로 선택된 token에는 영향을 주지만 보고되는 확률 목록에는
나타나지 않는다.

### 2.2 실측

probe 모델은 27B 원본이 아니라 같은 서버 계약을 확인하기 위한 소형 대체 모델이다.

| 항목 | 값 |
|---|---|
| 파일 | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| 출처 | <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF> |
| 크기 | 491,400,032 bytes |
| SHA-256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| 단일 token 후보 id | `" A"=362`, `" B"=425`, `" C"=356`, `" D"=422` |

평가기와 동일한 payload로 얻은 응답이다.

| prompt | 생성 token | `top_logprobs` id | `top_logprobs` piece | 동일성 검사 |
|---|---|---|---|---|
| `The capital city of France is` | `425` (`" B"`) | 12095, 1304, 30743, 32671 | `" Paris"`, `" __"`, `" ____"`, `" ______"` | 실패 |
| `Water boils at one hundred degrees` | `356` (`" C"`) | 13, 11, 518, 323 | `"."`, `","`, `" at"`, `" and"` | 실패 |

첫 사례의 `logprob`은 각각 -1.2412, -2.1606, -2.5331, -2.5401이고, 선택된 `" B"`의
`logprob`은 -6.7809이다. 선택된 token의 확률이 보고된 상위 4개보다 낮다는 사실 자체가
목록이 bias 이전 값이라는 직접 증거다.

두 관찰이 함께 성립한다.

- bias는 **선택**에 작동한다. 생성 token은 항상 네 후보 집합 안에 있었다.
- bias는 **보고되는 top-4**에 작동하지 않는다. 목록은 자연 상태의 상위 4개다.

따라서 사전등록 구현의 동일성 gate는 사실상 모든 문항에서 실패하며, 각 arm의 1번 문항에서
protocol failure로 중단된다. 이 gate로는 어떤 arm도 측정할 수 없다.

## 3단계: 대안 2를 기각한 이유

`post_sampling_probs: true`로 bias 적용 후 확률을 받는 경로를 검토하고 실측했다.

| 변형 | 보고 키 | top id | 생성 token | 4개 목록 |
|---|---|---|---|---|
| `post_sampling_probs=true`, `temperature=-1.0` | `top_probs` | 425 | 425 | 1개로 붕괴 |
| `post_sampling_probs=true`, `temperature=0.0` | `top_probs` | 425 | 425 | 1개로 붕괴 |
| `post_sampling_probs=true`, `temperature=1.0` | `top_probs` | 425, 362, 356, 422 | 362 | 4개 |
| `post_sampling_probs=true`, `temperature=1.0`, `top_k=0` | `top_probs` | 425, 362, 356, 422 | 425 | 4개 |

기각 사유는 세 가지다.

1. 키 이름이 `top_probs`로 바뀐다(`server-task.cpp:296`). 고정 commit에서 두 키를 모두
   받아들이면 응답 형식 변화가 조용히 다른 필드를 선택하게 되므로, 하나로 고정한 현재
   계약을 스스로 무너뜨린다.
2. greedy를 유지하는 `temperature <= 0`에서는 후보 목록이 1개로 붕괴한다. 4개 목록을 얻는
   유일한 설정은 `temperature > 0`이다.
3. `temperature = 1.0`은 선택을 확률적으로 만든다. 관측에서도 같은 prompt에 대해 한
   변형은 `" A"`, 다른 변형은 `" B"`를 생성했다. 이는 사전등록한 greedy Top-1 자체를
   깨뜨리므로, 진단 필드 하나를 얻기 위해 주 측정을 바꾸는 교환이 된다.

고정 commit에서 greedy Top-1과 4개짜리 post-bias 목록을 동시에 주는 설정은 없다. 요청을
바꾸지 않는 대안 1을 택한다.

## 4단계: 실제 변경

`src/mixstq/eval_llama_server.py`

- `score_completion`에서 `sorted(top_ids) == sorted(letter_ids)` 동일성 gate를 제거한다.
- `top_logprobs` 키 요구는 유지한다. 없거나 list가 아니면 키 이름을 명시하며 즉시 실패한다.
- 목록 각 원소의 정수 `id` 요구도 유지한다. `logprob`은 숫자 또는 null만 허용하며, 고정
  서버가 `-inf`를 null로 직렬화하므로 유한하지 않은 값은 null로 기록한다.
- 항목 record의 필드 이름을 의미가 드러나도록 `pre_sampling_top_ids`,
  `pre_sampling_top_logprobs`로 바꾸고, 반환값을 그대로 보존한다. 두 필드는 진단값이며
  판정에 쓰지 않는다는 주석을 코드에 남긴다.

유지되는 gate는 다음과 같다.

- `/tokenize`로 `A`, `B`, `C`, `D`가 서로 다른 단일 token인지 먼저 확인한다.
- 응답의 생성 token은 정확히 1개이며 후보 집합 안에 있어야 한다. 벗어나면 오답으로
  바꾸지 않고 protocol failure로 중단한다.
- `completion_probabilities`는 정확히 1개 항목이어야 한다.
- 응답 text와 선택 token의 letter가 일치해야 한다.
- 800개 record의 schema/provenance 검증, 원자적 진행 기록, resume provenance 일치,
  완료 결과 덮어쓰기 거부는 그대로다.

`tests/test_eval_llama_server.py`

- top-4 동일성 거부 test를 제거하고, 자연 상태 top-4가 후보 집합과 달라도 수용되며 record에
  그대로 기록되는 test로 바꾼다. fake 서버의 기본 응답도 probe에서 관측한 실제 형태
  (`" Paris"`, `" __"`, `" ____"`, `" ______"`)로 바꾼다.
- `top_logprobs` 키 누락 거부, 후보 외 생성 token 거부 test는 유지한다.
- null `logprob` 기록과 후보 집합과 동일한 top-4의 수용을 각각 추가로 검증한다.

## 5단계: 개정 시점 선언

이 개정은 Vast.ai 인스턴스를 임대하기 전, BF16·IQ3_XXS·IQ4_XS·Q4_K_M·Q5_K_M GGUF를
하나도 만들지 않은 시점, 어떤 arm의 800문항 점수·PPL·처리량도 관찰하지 않은 시점에
이루어졌다. 따라서 어떤 결과값도 이 결정에 영향을 줄 수 없었다. 개정 근거는 고정 commit의
소스와 27B와 무관한 소형 probe 모델의 응답 형식뿐이며, 두 증거 모두 정확도 수치를 포함하지
않는다.

측정을 시작한 뒤에는 이 계약을 다시 바꾸지 않는다. 실행 중 다른 응답 형식이 관찰되면 그
arm을 protocol failure로 중단하고 기록하며, 답을 보정하거나 gate를 완화하지 않는다.
