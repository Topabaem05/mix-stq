# MIX-STQ v0.26 사전등록: Qwen3.8-27B BF16 800문항 비열등성 재검

**사전등록일** 2026-08-31  
**상태** 결과 미열람, 실행 전 고정  
**주 질문** 참조 제약 IQ3_XXS로 MLP 192개를 3.0625 bpw로 재구성했을 때,
dense BF16 대비 Top-1 정확도 손실의 보수적 95% CI 상한이 2.0%p 이하인가?

이 문서는 결과 파일을 만들기 전에 표본, 모델, 양자화 범위와 판정 규칙을 고정한다.
v22와 v25의 200문항은 MMLU parquet의 앞부분을 그대로 사용해 57과목 중 일부 과목에
편중되었으므로 탐색 결과로만 남긴다. 이번 실행은 그 편향과 FP16/BF16 표기 오류를
동시에 바로잡는다.

## 1단계: 대상과 가설을 고정

| 항목 | 사전 고정값 |
|---|---|
| 모델 | `Qwen/Qwen3.8-27B` |
| 모델 revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| dtype | `torch.bfloat16` |
| 기준 팔 | `dense` |
| 시험 팔 | `dense_iq3_ref` |
| 양자화 범위 | language model 64층의 MLP `gate_proj`, `up_proj`, `down_proj`, 총 192 텐서 |
| 제외 범위 | attention, embedding, lm_head, vision tower, MTP head |
| 시험 형식 | 참조 제약 IQ3_XXS 재구성, 3.0625 bpw |
| importance matrix | `artifacts/qwen38_imatrix.pt`, SHA-256 `def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0` |
| 1차 지표 | 같은 800문항의 paired Top-1 correctness |
| 비열등 마진 | dense 우위 `+0.0200`, 즉 IQ3 손실 2.0%p |

이 실험의 범위는 MLP 재구성 효과다. 3.0625 bpw는 대상 MLP 텐서의 표현 비트이며,
모델 전체의 물리적 bpw 또는 GGUF 파일 bpw가 아니다.

## 2단계: 표본과 오염 방지를 고정

| 데이터셋 | revision | 선택 규칙 | 수량 |
|---|---|---|---:|
| `cais/mmlu`, config `all`, split `test` | `c30699e8356da336a370243923dbaf21066bb9fe` | 57과목 각각에서 source order상 처음 만나는 유효 4지선다 10개 | 570 |
| `allenai/ai2_arc`, config `ARC-Challenge`, split `test` | `210d026faf9955653af8916fad021475a3f00453` | source order상 처음 만나는 정답 포함 유효 4지선다 230개 | 230 |
| **합계** | | | **800** |

MMLU 과목명은 정렬해 결과에 기록하고 모든 과목이 정확히 10개인지 실행 전에 검사한다.
ARC 원본 test 1,172개 중 위 조건을 만족하는 항목은 사전 점검상 1,165개이므로 수량은
충분하다. 프롬프트와 정답을 포함한 정규화 item record를 순서대로 직렬화해 SHA-256
fingerprint를 만들며, dense와 IQ3 팔이 동일 fingerprint를 쓰지 않으면 실행을 실패시킨다.

정답 벡터 캐시는 최소한 다음 identity가 모두 일치할 때만 재사용한다.

- model id와 model revision
- dtype
- 두 dataset id/config/split/revision
- MMLU 과목당 수량과 ARC 수량
- ordered item fingerprint
- arm, low-layer 설정, 양자화 방법

길이만 같은 과거 캐시는 재사용하지 않는다. 실행 결과 JSON에도 같은 provenance와 각
문항의 task/subject를 보존한다.

## 3단계: 실행과 통계를 고정

Vast.ai의 단일 96 GB NVIDIA GPU에서 두 팔을 같은 코드 commit, tokenizer, prompt,
item order로 평가한다. 실행 전 `nvidia-smi`로 다른 compute process가 없음을 확인하고,
런타임의 GPU, torch, CUDA, transformers, datasets 버전과 실제 model parameter dtype을
결과에 기록한다.

통계는 문항별 correctness 차이 `dense - dense_iq3_ref`에 대해 계산한다.

- paired percentile bootstrap: 10,000회, seed 22, 보수적 양측 95% CI
- exact two-sided McNemar: dense-only correct와 IQ3-only correct의 불일치 쌍
- 전체 800문항이 1차 분석이며, MMLU/ARC 및 MMLU 과목별 값은 설명적 분석
- 중간 결과를 보고 표본, 마진, seed, 팔 또는 프롬프트를 바꾸지 않음

출력 목표는 `artifacts/qwen38_bf16_800.json`이다. 비정상 종료나 일부 문항만 끝난
파일은 최종 결과로 승격하지 않고, 실패 원인과 완료 수량을 별도 기록한다.

## 4단계: 판정 규칙을 고정

`delta = accuracy(dense) - accuracy(IQ3)`와 bootstrap CI `[L, U]`를 쓴다.

| 조건 | 판정 |
|---|---|
| `U <= +0.0200` | **2%p 마진 비열등성 통과** |
| `L > 0`, McNemar `p < 0.05`, 두 기준이 모두 성립 | **dense의 유의한 우위**, 성능 보존 실패 |
| `U > +0.0200`이나 유의 손실의 두 조건이 모두 성립하지 않음 | **미확정**, 보존 또는 손실을 주장하지 않음 |
| `U < 0`, McNemar `p < 0.05` | IQ3 우위 신호; 독립 재현 전 개선 주장 금지 |

점추정이 2%p 안에 있거나 McNemar가 유의하지 않다는 사실만으로 비열등성을 선언하지
않는다. 반대로 CI만 0을 배제하고 McNemar가 유의하지 않은 경우도 유의한 손실로 쓰지
않는다.

## 5단계: 결과 이후의 분기

1. 비열등성 통과 시 참조 IQ2_XXS 포팅과 IQ3 GGUF pack/C decoder parity를 진행한다.
2. 유의한 손실이면 3 bpw를 배포 후보로 승격하지 않고, 4–5 bpw의
   `IQ4_XS`, `Q4_K_M`, `Q5_K_M`으로 이동한다.
3. 미확정이면 불일치 쌍을 근거로 필요한 추가 표본 수를 다시 계산하되, 이번 800문항
   결과는 그대로 보존한다.
4. GGUF 왕복을 통과한 파일만 llama.cpp 속도·메모리와 Terminal Bench 2.1 full
   자유형식 평가 대상으로 인정한다.
5. 모든 실행 산출물은 SHA-256 manifest와 함께 Hugging Face에 보존하고, Vast.ai
   인스턴스는 산출물 회수 후 파괴하여 활성 목록 `[]`까지 확인한다.

이번 결과가 어느 분기로 가더라도 v22/v25의 과목 편향 200문항 결과를 소급해
“검증 표본”으로 승격하지 않는다.
