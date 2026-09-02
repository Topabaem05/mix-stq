# MIX-STQ v0.27 사전등록 개정 3: 보존 전송을 Mac 중계에서 호스트 직접 업로드로 바꾼다

**개정일** 2026-09-02
**대상 문서** [`mix-stq-v27-gguf-frontier-preregistered.md`](mix-stq-v27-gguf-frontier-preregistered.md) 5단계
**상태** 1차 유료 실행의 보존 단계가 대역폭으로 차단된 뒤, 어떤 양자화 arm의 Top-1·PPL·처리량도
관찰하기 전의 개정
**한 줄 변경** shard를 "trusted local machine(Mac)으로 내려 받아 로컬 write token으로 업로드"하던
경로를, "임대 호스트가 Hugging Face로 직접 업로드"하는 경로로 바꾼다. 업로드 대상 prefix, shard
크기 상한, 공개 재검증 의무, 전체 arm 검증 후 파기 규칙은 그대로 유지한다.

## 1단계: 원래 조항과 그것이 막힌 이유

5단계는 "한 arm씩 trusted local machine으로 전송하고, local write token으로
`topabaem/mix-stq-artifacts/paid-run/qwen38-gguf-frontier-v27/` 아래에 업로드한다"고 고정했다.
이 조항은 Mac이 62 GB를 받아서 다시 올릴 수 있다는 암묵적 가정 위에 있었다. 1차 실행에서 그
가정을 실측으로 반증했다.

| 구성 | 지속 처리량(실측) |
|---|---:|
| scp 단일 stream | 1.18 MB/s (60초 구간, 151 MB 전송) |
| ssh/dd 3 stream 병렬(위 scp와 동시) | 총 약 3.1 MB/s |
| ssh/dd 6 stream 병렬(단독) | 1.51 MB/s — 3 stream을 넘기면 오히려 악화 |

네 양자화 arm의 shard 총량은 62.05 GB(11.19 + 15.08 + 16.55 + 19.23)다. 최선으로 관측된
3 MB/s로도 내려받기만 5.7시간, 단일 stream 1.18 MB/s면 14.6시간이다. 같은 62 GB를 가정용
업링크로 다시 올려야 하고, 사전등록은 모든 arm의 공개 재검증이 끝날 때까지 Vast의 monolithic
GGUF를 유지하라고 요구하므로 그 왕복 내내 인스턴스가 시간당 $1.084로 과금된다. 당시 잔액은
$12.61(약 11.6시간)이었다. **왕복이 예산 안에 들어가지 않는다.** 실제로 전송은 151 MB에서
중단되었고, 업로드된 byte는 0이며 어떤 arm에도 공개 보존 marker가 없다.

두 번째 차단 요인도 같이 기록한다. Mac에 설치된 Hugging Face credential은 `token_role = read`
였고 업로드는 `403 Forbidden: you must use a write token to upload to a repository`로 실패했다.
credential 값은 읽거나 출력하거나 복사하지 않았다.

## 2단계: 개정된 보존 프로토콜

### 2.1 전송 경로

임대 호스트가 Hugging Face Hub로 **직접** 업로드한다. 모델 shard는 Mac을 경유하지 않는다.
Mac은 (a) 작은 증거 artifact 회수와 (b) 아래 2.4의 독립 확인만 담당한다.

### 2.2 write token 설치 — 사용자가 직접 수행한다

- **사용자 본인이** ssh로 호스트에 접속해 `huggingface-cli login`으로 short-lived
  **WRITE** token을 설치한다. 실행할 한 줄은 아래 4단계에 있다.
- 에이전트/오케스트레이터는 token 값을 **취급하지 않는다.** 읽지 않고, 출력하지 않고,
  복사하지 않고, 어떤 파일에도 쓰지 않는다.
- token 값은 저장소, 명령 인자(argv), 로그, planner 출력, artifact 어디에도 존재하지 않는다.
  planner는 credential이 들어갈 수 없는 argv만 생성하며, 이는 test로 강제한다
  (`tests/test_gguf_run_plan.py`).
- 업로드 명령은 호스트에 **이미 로그인된** CLI의 자격 증명을 사용한다. 명령줄에 `--token`을
  주지 않는다.
- token은 이번 실행 범위로만 발급하고, 실행 종료 시 **사용자가 직접 폐기(revoke)** 한다.
  인스턴스 destroy는 폐기를 대신하지 않는다.

역할 확인은 값 노출 없이 `hf auth whoami`(또는 `huggingface-cli whoami`) 출력의 role 표시로만
한다.

### 2.3 업로드 대상 — 사전등록 prefix 그대로

공개 dataset 저장소 `topabaem/mix-stq-artifacts`, 사전등록된 prefix
`paid-run/qwen38-gguf-frontier-v27/` 를 **그대로** 쓴다.

```
paid-run/qwen38-gguf-frontier-v27/IQ3_XXS/     shard 파일
paid-run/qwen38-gguf-frontier-v27/IQ4_XS/      shard 파일
paid-run/qwen38-gguf-frontier-v27/Q4_K_M/      shard 파일
paid-run/qwen38-gguf-frontier-v27/Q5_K_M/      shard 파일
paid-run/qwen38-gguf-frontier-v27/projector/   vision mmproj (text bpw 분모에서 제외)
paid-run/qwen38-gguf-frontier-v27/evidence/    작은 artifact(calibration, imatrix, smoke, preflight)
```

1차 실행에서 planner 없이 진행되던 중 coordinator 지시로 `qwen38-gguf-v27/` prefix가 거론된
적이 있으나, **그 경로에는 아무것도 기록되지 않았다.** 업로드 byte가 0이었으므로 어떤 경로도
실제로 사용되지 않았고, 이 개정으로 사전등록 prefix 하나만 남는다.

### 2.4 공개 재검증 — 의무는 그대로, 수행 위치만 바꾼다

"업로드 후 public unauthenticated re-download로 모든 shard SHA-256과 aggregate manifest를 다시
확인한다"는 조항은 **완화하지 않는다.** 62 GB를 Mac으로 다시 내려받는 것이 불가능하므로 수행
위치를 바꾼다.

**채택(가장 엄격하면서 실행 가능한 안):** 모든 shard의 **완전 unauthenticated 재다운로드와
sha256 대조를 호스트에서** 수행한다. 각 검증 다운로드는 credential 환경변수를 제거한 **새
프로세스**에서 실행한다. 구체적으로 `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` 등 이름에 TOKEN이
들어가는 모든 변수와 `HF_`로 시작하는 모든 변수(`HF_HUB_OFFLINE` 포함)를 제거한 뒤 CLI를
exec한다. 재다운로드본은 shard 단위로 대조 직후 삭제하므로 추가 disk 최대 점유는 shard 하나
(≤ 8 GiB)이며, Task 5 종료 시점 여유 공간 150 GB 안에서 안전하다. 호스트의 실측 하향 대역폭은
4,668 Mbps였고 검증 트래픽은 호스트↔HF 구간에만 발생하므로, 62 GB 재다운로드는 예산 안에서
끝난다. 이 방식은 사용자가 허용한 최소선(“Mac에서 LFS pointer sha256 확인 + arm당 최소 1개
shard의 호스트 전체 재다운로드”)보다 엄격하다.

**보강(독립 지점 확인):** Mac에서 각 shard의 LFS pointer sha256을 HTTP metadata로 조회해
로컬 manifest의 sha256과 대조한다. 이 확인은 호스트와 다른 네트워크 경로·다른 기계에서
이루어지므로, 호스트 단독 검증이 놓칠 수 있는 "호스트에 남아 있는 캐시를 검증했을 뿐"인
실패 양식을 배제한다. Mac 쪽 확인 역시 unauthenticated 요청으로 한다.

두 확인이 모두 통과한 shard만 검증된 것으로 본다.

### 2.5 파기 규칙

- **네 양자화 arm 전부**가 공개 marker(업로드 완료 + 2.4의 두 확인 통과)를 가진 뒤에만
  인스턴스를 destroy한다. arm 하나라도 미검증이면 파기하지 않는다.
- BF16 monolith(53,808,281,952 byte, sha256
  `03ab7ad49486af2f111ed8d7616a0f485f9c5032bd6bc2419b84bb3b90f3930f`)는 **업로드하지 않는다.**
  고정 model revision과 고정 converter argv로 재생성 가능하며, 재생성 identity를 위해 SHA와
  byte size를 manifest에 기록하는 것으로 보존 의무를 대신한다. 이는 사전등록 5단계의
  "BF16 GGUF는 원본 model revision과 재생성 manifest를 보존한다"는 조항과 같은 취지다.
- 네 양자화 arm, calibration, imatrix, 로그, 결과를 우선 영구 보존한다는 우선순위는 그대로다.

### 2.6 유지되는 조항

shard당 8 GiB 상한(`llama-gguf-split --split --split-max-size 8G`), 공개 저장소와 prefix,
모든 shard의 sha256 대조, 전체 arm 검증 전 원본 파기 금지, token을 호스트 파일·환경·argv·로그에
남기지 않는다는 요구는 그대로 유지된다. 마지막 항목의 의미만 명확히 한다. 금지 대상은 token
**값**의 노출이며, 사용자가 대화형으로 설치해 CLI가 자체 credential 저장소에 보관하는 것은
허용된다. 이전 문구가 사실상 배제하던 것은 후자였고, 그것이 유일하게 실행 가능한 경로다.

## 3단계: 개정 시점 선언

이 개정은 1차 유료 실행의 Task 5 산출물 생성과 BF16 arm의 800문항 Top-1 696/800 관찰 이후,
그러나 **어떤 양자화 arm의 Top-1, 어떤 arm의 PPL, 어떤 arm의 llama-bench 값도 관찰하기 전에**
이루어졌다. 이 개정은 측정 방법·표본·arm 집합·판정 규칙을 전혀 건드리지 않고 artifact 전송
경로만 바꾸므로, 어떤 결과값도 이 결정에 영향을 줄 수 없고 이 결정도 어떤 결과값에 영향을 줄
수 없다. 근거는 전부 대역폭 실측과 credential 권한이라는 운영 사실이다.

## 4단계: 사용자가 실행할 한 줄

호스트/포트는 실제 인스턴스 값으로 바꾼다. 에이전트는 이 명령을 실행하지 않으며 token 값을
보지 않는다.

```
ssh -t -p <PORT> root@<HOST> '/workspace/mix-stq/artifacts/qwen38-gguf-v27/venv/bin/huggingface-cli login'
```

- `-t`가 TTY를 붙이므로 token 입력이 화면에 표시되지 않고 shell history에도 남지 않는다.
- token을 `ssh ... echo <token>` 형태로 전달하지 않는다. 그러면 argv와 history에 남는다.
- 설치 후 값 노출 없이 역할만 확인한다.

```
ssh -p <PORT> root@<HOST> '/workspace/mix-stq/artifacts/qwen38-gguf-v27/venv/bin/hf auth whoami'
```

- 실행 종료 시 사용자가 Hugging Face 설정에서 해당 token을 폐기하고, 폐기 사실을 ledger에
  기록한다.
