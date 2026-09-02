# MIX-STQ v0.27 사전등록 개정 2: imatrix chunk 수의 의미와 PPL corpus 직렬화 고정

**개정일** 2026-09-02
**대상 문서** [`mix-stq-v27-gguf-frontier-preregistered.md`](mix-stq-v27-gguf-frontier-preregistered.md) 2단계, 4.2
**상태** 1차 유료 실행의 Task 5 파일 생성과 BF16 Top-1 관찰 이후, 어떤 양자화 arm의
Top-1·PPL·처리량도 관찰하기 전의 개정
**한 줄 변경** (1) "정확히 128 chunks"는 `--chunks 128` argv 값을 상한으로 고정한다는 뜻이며
실현 chunk 수는 `floor(tokens/512)`로 산출되어 반드시 로그에 기록한다. (2) held-out PPL corpus의
직렬화 규칙을 1차 실행에서 실제로 사용한 형태 그대로 고정한다.

두 항목 모두 측정 규칙의 완화가 아니라 문구와 직렬화의 명시다. 고정된 모델 revision,
llama.cpp commit, corpus, imatrix 설정값, arm 집합, 800문항 표본과 판정 규칙은 바꾸지 않는다.

## 1단계: imatrix chunk 수

### 1.1 사전등록이 말한 것

2단계는 `llama-imatrix` 고정값을 "context 512, batch 512, ubatch 128, 정확히 128 chunks,
threads 32, `--no-ppl`, GGUF output"으로 적었다. "정확히 128 chunks"라는 표현은 argv 값과
실현값을 구분하지 않았다.

### 1.2 실행에서 관찰한 것

1차 유료 실행(instance `49615861`, 호스트 root `artifacts/qwen38-gguf-v27`)에서 imatrix는
고정 argv를 **한 글자도 바꾸지 않고** 실행했다.

```
llama-imatrix --model models/qwen38-27b-bf16.gguf --file calibration/corpus.txt \
  --output-file imatrix/qwen38-27b.imatrix.gguf \
  --ctx-size 512 --batch-size 512 --ubatch-size 128 --chunks 128 --threads 32 --no-ppl
```

실행 직전 통과한 `--action token-preflight` gate(고정 `llama-tokenize --ids --show-count`)의
출력은 다음과 같다.

```json
{"capacity_tokens": 65536, "chunks": 128, "committed": true,
 "corpus_sha256": "79f0c5cf125b9da642e82519e8630885c67c75336dd628eba69a898cdac681d5",
 "token_count": 10523, "tokens_per_chunk": 512}
```

llama.cpp는 `compute_imatrix: computing over 20 chunks, n_ctx=512, batch_size=512, n_seq=1`을
로그에 남겼다. 고정 corpus는 46,981 byte / 10,523 token이므로 512 token짜리 chunk를 20개만
채운다.

| 항목 | 값 |
|---|---|
| corpus sha256 | `79f0c5cf125b9da642e82519e8630885c67c75336dd628eba69a898cdac681d5` |
| corpus bytes | 46,981 (상한 65,536) |
| 측정 token 수 | 10,523 |
| argv `--chunks` | 128 |
| 실현 chunk 수 | 20 |
| imatrix sha256 | `e9656bbc8f3699b47b1c4f0c75323721a0ef6409d8dc338473c69fb069fee1c8` |
| imatrix bytes | 13,642,688 |

### 1.3 개정 내용

- 고정값은 argv `--chunks 128` **그대로**다. 이 값은 상한(ceiling)이며, 이 상한을 바꾸는 것은
  계약 위반이다.
- 실현 chunk 수는 `floor(token_count / 512)`로 결정되며, arm manifest와 실행 로그에 **반드시**
  기록한다. 1차 실행의 실현값은 20이다.
- 저장소 gate는 이미 이 의미로 구현되어 있다. `src/mixstq/gguf_run_plan.py`의
  `exact_tokenizer_preflight`는 `0 < token_count <= IMATRIX_CHUNKS * IMATRIX_CONTEXT_TOKENS`
  (= `0 < token_count <= 128 * 512 = 65,536`)를 요구한다. 즉 128은 코드에서 처음부터 용량
  상한이었고, 하한이나 등식이 아니었다.
- 실현 128 chunk를 강제하려면 고정 corpus(65,536 byte 상한, record당 512 byte 하한 규칙) 또는
  고정 imatrix 설정 중 하나를 바꿔야 한다. 두 가지 모두 실제 계약 위반이므로 하지 않는다.

네 양자화 arm은 모두 위 단일 imatrix 파일 하나를 공유한다는 조항은 그대로다.

## 2단계: held-out PPL corpus 직렬화

### 2.1 사전등록이 말한 것

4.2는 "perplexity는 calibration과 겹치지 않는 pinned WikiText-2 `test` split을 source order로
정규화한 별도 corpus에서 측정한다"고 적었다. 소스, split, 순서, 정규화 여부는 고정했지만
**정규화를 record마다 적용하는지 연결 후 한 번 적용하는지**, record를 무엇으로 잇는지는
적지 않았다. 같은 서술로도 서로 다른 byte 열이 나올 수 있다.

### 2.2 1차 실행에서 실제로 만든 corpus

1차 실행에서 호스트에 만들어 두었고 로컬 증거로 보존한 파일의 생성 규칙을 그대로 고정한다.

| 항목 | 고정값 |
|---|---|
| dataset | `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test` |
| revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| field | `text` |
| record 수 | 4,358 (필터 없이 split 전체) |
| 순서 | dataset source order |
| 직렬화 | 4,358개 record 문자열을 그 순서대로 **연결**한 뒤 |
| 정규화 | 연결된 문자열 **전체에 한 번** `mixstq.llama_calibration.normalize_text` 적용 |
| bytes / chars | 1,284,763 UTF-8 / 1,282,729 |
| sha256 | `03492eaf99762251b0c9ed3bc4229294e7f3a03c5ec8cb9cdb61f54999539e11` |

byte 열을 결정하는 두 줄은 다음과 같다.

```python
raw = "".join(record["text"] for record in dataset)   # source order, 구분자 없음
text = normalize_text(raw)                            # 연결된 전체에 1회
```

추가 separator를 넣지 않는다(record 사이에 어떤 구분자도 삽입하지 않는다). record별 정규화
후 연결하는 순서가 아니라, **연결 후 1회 정규화**가 고정 규칙이다. calibration corpus가
쓰는 `\n\n\x1e\n\n` separator는 PPL corpus에 적용되지 않는다.

위 sha256은 서로 독립적인 두 지점에서 일치했다. 생성 직후 호스트에서 계산한 값과,
`final-artifacts.tar.gz`(sha256 `372b687664981ba4471d40d66e858e16508d4371f9100d644a7d8eb053b40bf0`)를
Mac으로 회수해 풀고 다시 계산한 `evidence/wikitext/wikitext2-test.txt`의 값이 같다.

### 2.3 개정 내용

- 다음 실행은 이 파일을 **재생성하지 않고** 보존본을 그대로 호스트로 복사해 쓴다. 재생성할
  경우 위 규칙으로 만들고 sha256이 `03492eaf…39e11`과 같은지 확인한 뒤에만 사용한다.
- PPL은 계속 고정 `llama-perplexity` binary, 동일 context/chunk 설정, 동일 GPU offload로
  측정하며 absolute PPL과 BF16 대비 상대 증가율을 함께 기록한다. 이 조항은 바뀌지 않는다.
- calibration(WikiText-2 `train`)과 PPL(WikiText-2 `test`)이 같은 revision의 서로 다른 split이라는
  held-out 조건도 그대로다.

## 3단계: 개정 시점 선언

이 개정은 **Task 5 파일 생성(BF16 GGUF, imatrix, 네 양자화 GGUF, smoke, audit, split)과
BF16 arm의 800문항 Top-1 696/800 관찰 이후**, 그리고 **어떤 양자화 arm의 Top-1, 어떤 arm의
PPL, 어떤 arm의 llama-bench 처리량도 관찰하기 전**에 이루어졌다. 1차 실행에서 IQ3_XXS,
IQ4_XS, Q4_K_M, Q5_K_M의 Top-1은 모두 미측정이고(IQ3_XXS는 49문항 부분 진행 기록만 남았으며
점수로 집계되지 않았다), 다섯 arm 전부의 PPL과 llama-bench도 미측정이며, paired bootstrap CI,
McNemar p, 2%p 비열등 판정도 산출되지 않았다.

따라서 이 개정이 바꾼 두 항목은 어떤 **비교 결과**로도 영향을 받을 수 없었다. 두 항목 모두
BF16 단독 값이나 arm 간 차이와 무관한, argv 의미의 명시와 입력 byte 열의 명시다. 관찰된 유일한
수치인 BF16 696/800은 imatrix chunk 수와 무관하고(BF16 arm은 imatrix를 쓰지 않는다) PPL corpus
직렬화와도 무관하다(BF16 PPL은 측정되지 않았다).

측정을 재개한 뒤에는 이 계약을 다시 바꾸지 않는다. 실현 chunk 수가 20이 아닌 값으로 관찰되면
그 값을 그대로 기록하고 원인을 보고하며, corpus나 argv를 사후에 맞추지 않는다.
