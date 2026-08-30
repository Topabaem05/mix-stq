# MIX-STQ v1.5 — 저장소 결정 및 Qwen3.8-27B 분석

**작성일** 2026-08-30
**GPU 지출** $0 (HTTP Range read + CPU)

---

## 1. 저장소 결정: 별도 프로젝트

MIX-STQ를 Pacific에 커밋하지 않고 **신규 저장소 `Topabaem05/mix-stq`**로 분리할 것을 권합니다.

| 근거 | 내용 |
|---|---|
| 스택 불일치 | Pacific = Python/PyTorch/safetensors. MIX-STQ = C/ggml/GGUF |
| 불변성 계약 충돌 | Pacific은 535개 파일이 SHA-256 고정, 테스트가 바이트 단위 검증 |
| Pacific 자체 원칙 | "새 codec은 새 ID로 등록, 기존 codec 불변" — GGUF 경로는 이 범위를 넘음 |
| 의존성 방향 | MIX-STQ는 llama.cpp 생태계(ggml 타입, imatrix, GGUF)에 결합 |
| 현재 상태 | 코드가 이미 Pacific 트리 밖(`work/mixstq/`)에 있음 |

Pacific에는 `docs/research/`에 참조 한 줄만 추가하면 충분합니다. 연구 계보상
Pacific의 저비트 작업에서 파생됐으므로 추적 가치는 있습니다.

**주의**: 현재 Pacific clone에는 리포 정리 변경 507개가 push되지 않은 상태입니다.

## 2. Qwen3.8-27B 구조

| 항목 | 값 |
|---|---|
| 파라미터 | 27,781,427,952 |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (dense, MoE 아님) |
| 레이어 | 64 (full_attention 16 + linear_attention 48) |
| hidden / intermediate | 5120 / 17408 |
| head_dim | 256, GQA 24:4 |
| vocab | 248,320 |

제 MIX-STQ 검증 대상(OLMoE, Qwen1.5-MoE)과 달리 **dense hybrid attention**입니다.

## 3. 양자화별 크기와 bpw (측정)

파일 크기는 unsloth 공식 GGUF 저장소 API, bpw는 위 파라미터 수로 계산했습니다.

| 양자화 | 크기 | bpw | 압축 |
|---|---:|---:|---:|
| BF16 | 54.66 GB | 15.740 | 1.00x |
| Q8_0 | 29.05 GB | 8.365 | 1.88x |
| UD-Q6_K | 21.98 GB | 6.329 | 2.49x |
| UD-Q5_K_XL | 20.88 GB | 6.013 | 2.62x |
| UD-Q5_K_M | 19.77 GB | 5.693 | 2.76x |
| UD-Q4_K_XL | 17.56 GB | 5.057 | 3.11x |
| UD-Q4_K_M | 16.46 GB | 4.740 | 3.32x |
| UD-IQ4_XS | 14.25 GB | 4.103 | 3.84x |
| UD-IQ3_S | 12.04 GB | 3.467 | 4.54x |
| UD-IQ2_S | 8.37 GB | 2.410 | 6.53x |
| UD-IQ2_XXS | 7.27 GB | 2.093 | 7.52x |
| UD-IQ1_M | 6.73 GB | 1.938 | 8.12x |
| UD-IQ1_S | 6.19 GB | 1.782 | 8.83x |

## 4. 품질 손실 (외부 출처, 제 측정 아님)

| 양자화 | KL divergence | top-1 일치 | Q8 대비 손실 |
|---|---:|---:|---:|
| Q8_0 | 0.00064 | 98.92% | 기준점 |
| UD-Q6_K | 0.00107 | 98.67% | ~0.25 pp |
| UD-Q5_K_M | 0.00419 | 97.34% | ~1.6 pp |
| UD-Q4_K_XL | 0.00955 | 96.02% | ~2.9 pp |
| UD-Q4_K_M | 0.01126 | 95.59% | ~3.3 pp |
| UD-IQ4_XS | 0.01248 | 95.39% | ~3.5 pp |
| UD-IQ3_S | 0.03247 | 92.41% | ~6.5 pp |
| UD-IQ2_S | 0.09832 | 87.18% | ~11.7 pp |
| UD-IQ2_XXS | 0.25663 | 79.44% | ~19.5 pp |
| UD-IQ1_M | 0.34212 | 76.34% | ~22.6 pp |

Wikitext-2 perplexity: Q8_0 6.9557 / Q4_K_M 6.9576 (+0.03%) / IQ4_XS 7.0130 / IQ3_XXS 7.2441.

태스크 벤치마크: Q4_K_M은 GPQA Diamond·IFBench·Terminal-Bench 2.1에서 BF16과 실질 구별 불가.
Q2_K_XL은 저하되나 사용 가능. **1-bit는 일부 태스크에서 무작위 추측 수준 붕괴.**

출처: AtomicChat GGUF 카드, kingy.ai, r/LocalLLaMA, r/LocalLLM, Quesma 블로그.
파일 크기가 출처마다 다른 것은 빌드 차이입니다.

### KL이 태스크 붕괴를 예측하지 못합니다

1-bit의 top-1은 76%로 "그럭저럭"인데 태스크는 붕괴합니다. arXiv:2608.06564이 보고한
margin 붕괴(2비트에서 배율 중앙값 0.00)와 정확히 같은 현상입니다.
**KL·top-1만 보는 검증은 이 실패를 놓칩니다.** 제 검증 계약이 margin을 필수로 두는 근거입니다.

### 무릎 위치가 제 관측과 일치

IQ3_S(3.47 bpw) 92.4% → IQ2_S(2.41) 87.2% → IQ2_XXS(2.09) 79.4%.
Pacific E-ladder가 35B-A3B에서 측정한 "무릎은 3.5비트 바로 위"와 같은 형태입니다.

## 5. LTC 전제가 dense 모델에서도 성립

Qwen3.8-27B는 MoE가 아니므로 전제를 직접 측정했습니다(dense MLP gate/up 텐서 8개, 각 65,536 표본).

| 모델 | 구조 | 자연 0 비율 | 텐서 간 편차 | kurtosis |
|---|---|---:|---:|---:|
| OLMoE-1B-7B | MoE expert | 0.313 | 4.4% | ~3.0 |
| Qwen1.5-MoE | MoE routed | 0.312 | 2.2% | 3.0–3.5 |
| Qwen1.5-MoE | MoE shared | 0.315 | 1.7% | 3.2–4.0 |
| **Qwen3.8-27B** | **dense MLP** | **0.312** | **2.0%** | **3.05–3.27** |

**세 아키텍처가 0.312–0.315로 동일하며, dense 모델에서도 성립합니다.**
자연 0 비율 ~0.31은 MoE 특성이 아니라 **학습된 LLM 가중치의 일반 성질**로 보입니다.
3:4가 강제하는 0.250은 어느 모델에서도 부족합니다.

따라서 LTC는 Qwen3.8-27B 같은 dense hybrid-attention 모델에도 적용 가능성이 있습니다.
단, 품질 검증은 GPU가 필요하며 미수행입니다.

## 6. 검증 기록

| 주장 | 근거 | 상태 |
|---|---|---|
| 파라미터 27,781,427,952 | HF API `safetensors.total` | 관찰됨 |
| 레이어 64 = full 16 + linear 48 | config.json `layer_types` | 관찰됨 |
| 파일 크기 30종 | unsloth GGUF 저장소 API | 관찰됨 |
| bpw 값 | 크기 x 8 / 파라미터 수 | 계산됨 |
| dense MLP 자연 0 = 0.312 | 실제 BF16 텐서 8개 | 관찰됨 |
| **KL / top-1 / PPL / 태스크 정확도** | **외부 웹 출처** | **미측정** |

품질 표는 제가 측정한 것이 아닙니다. 서로 다른 측정자의 결과이며 빌드·프롬프트·표본이
통일되지 않았을 수 있습니다. 제 연구의 v1.0 결과처럼 쌍 부트스트랩 CI가 없습니다.
