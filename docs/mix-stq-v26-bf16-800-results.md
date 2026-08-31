# MIX-STQ v0.26 결과: Qwen3.8-27B 참조 IQ3_XXS가 2%p 비열등 마진을 통과했습니다

**실행일** 2026-08-31

**사전등록** [`mix-stq-v26-bf16-800-preregistered.md`](mix-stq-v26-bf16-800-preregistered.md)

**판정** `noninferior`

Qwen3.8-27B를 BF16로 불러와 language-model MLP 192개를 참조 제약
IQ3_XXS로 재구성한 800문항 결과는 dense **0.8700 (696/800)**,
IQ3 **0.86625 (693/800)**입니다. 사전 고정한 `dense - IQ3` 차이는
`+0.00375`이고 paired bootstrap 95% CI는 `[-0.0100, +0.0175]`입니다.
CI 상한 `+0.0175`가 비열등 마진 `+0.0200` 이내이므로 **2%p 마진 비열등성은
통과**했습니다. exact McNemar `p=0.7111`로 dense의 유의한 우위도 없습니다.

이 결론은 **MLP 재구성 하네스의 Top-1 정확도**에 한정됩니다. 3.0625 bpw는
대상 MLP payload의 비트 예산이지 모델 전체 또는 GGUF 파일의 물리 bpw가 아닙니다.
아직 packed GGUF, llama.cpp C 디코더, 자유형식 생성 또는 Terminal Bench 결과가
아닙니다.

## 1단계: 대상과 판정 규칙

결과를 보기 전에 다음 항목을 고정했습니다.

| 항목 | 고정값 |
|---|---|
| 모델 | `Qwen/Qwen3.8-27B` |
| 모델 revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| 실행 코드 | `92f86a5c8d3f1cca2e4eef8007b6c5c7ec9cce37` |
| dtype | `torch.bfloat16` |
| 표본 | MMLU 57과목 × 10 = 570, ARC-Challenge 230, 합계 800 |
| item fingerprint | `a72515282c6fc20f34188b3102d99468ab2b02266105ed9c6e4ec405fbad8fd0` |
| 팔 | `dense`, `dense_iq3_ref` |
| 양자화 범위 | 64층 × `gate_proj`, `up_proj`, `down_proj` = 192 MLP 텐서 |
| 제외 범위 | attention, embedding, lm_head, vision tower, MTP head |
| 1차 delta | `accuracy(dense) - accuracy(IQ3)` |
| 비열등 기준 | paired 95% CI 상한 `<= +0.0200` |
| 통계 | bootstrap 10,000회 seed 22, exact two-sided McNemar |

MMLU와 ARC dataset revision도 각각
`c30699e8356da336a370243923dbaf21066bb9fe`와
`210d026faf9955653af8916fad021475a3f00453`으로 고정했습니다. 이전 200문항
결과는 MMLU 과목 편향과 FP16/BF16 불일치가 있어 탐색 기록으로만 남깁니다.

## 2단계: 실행과 오염 방지

두 팔은 같은 단일 GPU, 같은 item order와 prompt, 같은 모델·데이터 revision에서
연속 실행했습니다.

| 실행 증거 | 관측값 |
|---|---|
| run id | `a0546c5ade144d2781f1fcf2ca8e9822` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB |
| 실행 전 compute process | 없음 |
| Python / torch / CUDA | 3.11.10 / 2.11.0+cu128 / 12.8 |
| transformers / datasets | 5.16.1 / 4.8.5 |
| 실제 floating parameter dtype | 전·후 모두 `torch.bfloat16`, 26,895,998,464 elements |
| 평가 시작·종료 | 09:10:21–09:51:59 UTC |
| 평가 명령 end-to-end wall time | **2,498초 (41분 38초)** |

strict protocol은 800개 fingerprint, BF16 분포, 192개 tensor inventory, skip 0,
모델·데이터 revision과 importance-matrix 해시 중 하나라도 다르면 실행 또는 캐시
재사용을 거부합니다. 완료 후 결과 JSON, 완료 마커와 correctness vector를 별도
스크립트로 다시 읽어 bootstrap과 McNemar를 독립 재계산했습니다.

첫 Vast 인스턴스는 SSH-ready 상태가 되지 않아 GPU 실행 전에 폐기했습니다. 두 번째
인스턴스에서만 본 평가를 수행했으며, `/usr/bin/time` 부재도 evaluator 시작 전에
검출해 start/end epoch 기록으로 바꿨습니다. 두 사건 모두 결과 파일 생성 이전이라
팔 또는 표본을 바꾸지 않았습니다.

## 3단계: 측정 결과

| 팔 | 대상 MLP bpw | mean error | Top-1 | dense 대비 |
|---|---:|---:|---:|---:|
| dense BF16 | 16 | — | **0.8700 (696/800)** | 기준 |
| 참조 IQ3_XXS | **3.0625** | 0.027723 | **0.86625 (693/800)** | **−0.00375** |

| 비교 | dense−IQ3 delta | paired 95% CI | dense-only / IQ3-only | McNemar p | 판정 |
|---|---:|---|---:|---:|---|
| 전체 800 | **+0.00375** | **[−0.0100, +0.0175]** | 16 / 13 | 0.7111 | 2%p 비열등 통과, 유의 차이 없음 |

설명적 태스크 분해에서는 MMLU가 두 팔 모두 476/570이고, ARC-Challenge가 dense
220/230 대 IQ3 217/230입니다. 1차 판정은 이 사후 분해가 아니라 사전 고정한 전체
800문항 paired 통계입니다.

양자화 계획은 정확히 17,112,760,320개 MLP weight를 처리했고 skip은 0입니다.
참조 형식의 이론 payload는 6,550,978,560바이트, 3.0625 bpw입니다. 현재 평가기는
참조 값으로 재구성한 뒤 BF16 tensor로 추론하므로, 이 수치는 **실제로 저장한 파일
크기**가 아닙니다.

## 4단계: 판정, 보존과 한계

CI 상한 `+1.75%p`가 사전 마진 `+2.00%p`보다 작아 `noninferior` 분기로 갑니다.
이 결과는 “정확히 차이가 없다”는 뜻이 아닙니다. 현재 데이터는 IQ3가 1.0%p 더
좋은 경우부터 dense가 1.75%p 더 좋은 경우까지와 양립합니다. 다만 이번 하네스와
범위에서 2%p를 넘는 손실은 95% CI 밖입니다.

| 증거 | SHA-256 / 상태 |
|---|---|
| 결과 JSON | `b9e90b1bb0aa337fd5937f1bf98e6e08918579d44ab3c9aa0a7190347cc7a9fe` |
| 32-file manifest | `5a891351e7a3a55651586db16bacee732df54f3528ab5ea1bc9ba07df3f6548a` |
| importance matrix | `def82108b5d58871434cfeb87009eee8e7b8c68b6c4eb9512ffffa4f9ca2a9e0` |
| Hugging Face 보존 | [`paid-run/qwen38-bf16-800`](https://huggingface.co/datasets/topabaem/mix-stq-artifacts/tree/main/paid-run/qwen38-bf16-800) |
| HF 보존 commit | [`2e7a5a64`](https://huggingface.co/datasets/topabaem/mix-stq-artifacts/commit/2e7a5a64b284d00e3d262a8542e94194e3f14fe1) |
| Vast 종료 (operator 관측) | instance `49369787` 파괴 후 활성 목록 `[]`, active burn `$0.0000` |

모든 32개 파일은 로컬에서 manifest 검증 후 HF에 올렸고, 공개 경로에서 새 임시
디렉터리로 다시 내려받아 같은 해시를 확인한 뒤 인스턴스를 종료했습니다.

비용은 정상 인스턴스 종료 직전 API 관측 `$0.8658`과 SSH-ready 실패 인스턴스
`$0.0532`를 합친 **$0.9190**입니다. Vast 종료 상태와 비용은 이 운영 세션에서
관측한 값이며 32-file 실행 manifest에는 API snapshot이 포함되지 않았습니다. 따라서
보존 산출물만으로 독립 재계산할 수 없고, 별도 청구서 반올림과 몇 센트 차이가 날 수
있습니다.

남은 한계는 명확합니다.

- attention, embedding, lm_head 등 모델 전체를 양자화하지 않았습니다.
- packed IQ3_XXS block과 GGUF 파일을 만들지 않았고 C decoder parity도 없습니다.
- PyTorch letter-logprob 객관식 하네스이며 자유형식 생성 성능이 아닙니다.
- llama.cpp 처리량·메모리와 Terminal Bench 2.1 full을 측정하지 않았습니다.
- 참조 IQ2_XXS와 4–5 bpw 대조군은 아직 같은 증거 수준으로 실행하지 않았습니다.

## 5단계: 다음 연구 분기

비열등성 통과에 따라 연구 우선순위를 다음처럼 고정합니다.

1. **IQ3 GGUF/C parity**: 참조 IQ3_XXS를 packed block으로 쓰고 Python oracle,
   GGUF round-trip, llama.cpp C decoder 출력이 같은지 확인합니다.
2. **전체 모델·실제 bpw**: 고정한 llama.cpp commit으로 Qwen3.8-27B 전체 GGUF를
   만들고 파일 바이트, 전체 parameter 기준 bpw, 양자화 tensor inventory를 기록합니다.
3. **4–5 bpw 대조군**: 같은 원본 revision에서 `IQ4_XS`, `Q4_K_M`, `Q5_K_M`을
   변환합니다. 포맷 이름이 아니라 실제 파일 bpw로 차트에 놓습니다.
4. **동일 하네스 비교**: IQ3/4/5 GGUF 모두에 같은 800문항 Top-1, held-out
   perplexity, `llama-bench` tok/s·RAM·VRAM을 적용합니다. 참조 IQ2_XXS 포팅은
   3 bpw 아래 과학적 경계를 확인하는 별도 팔로 유지합니다.
5. **자유형식 full 평가**: GGUF round-trip과 C parity를 통과한 파일만
   `topabaem@100.73.38.99`로 보내 Claude Code 하네스와 llama.cpp로
   Terminal Bench 2.1 full을 실행합니다. 각 arm의 완료율·점수·wall time을 따로
   기록하며 partial 결과를 full과 합치지 않습니다.

Qwen1.5-MoE 교차 확인은 이 배포 게이트가 양성일 때까지 미룹니다. 다음 차트에는
현재 측정된 BF16와 IQ3만 표시하고, 4–5 bpw 값은 실제 측정이 끝난 뒤 같은 축에
추가합니다.
