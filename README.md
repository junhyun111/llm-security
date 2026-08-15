# LLM Security Conditional Expert Pipeline

## 웹 기반 프로젝트 분석 및 승인형 수정

로컬 웹에서 프로젝트 폴더 전체를 선택하면 업로드 직후 분석 작업이 시작됩니다.
정적 분석과 Router는 로컬에서 실행되고, Router가 선택한 여러 Expert 작업은
`OPENROUTER_EXPERT_MODEL`의 단일 요청으로 묶입니다. 응답은 다시 Expert별 Finding으로
분리되며, 정적 근거 검증에는 LLM을 추가 호출하지 않습니다. 결과 화면에서 취약점
위치, CWE, 근본 원인과 근거를 확인할 수 있습니다.

검증된 Finding 여러 개를 선택하면 한 번의 통합 패치 요청으로 coordinated diff를
생성합니다. 사용자가 diff를 다시 승인해야만 별도의 프로젝트 복사본에 적용되며,
업로드 원본은 변경되지 않습니다. 따라서 정상 흐름의 OpenRouter 호출 예산은
프로젝트당 탐지 1회와 통합 패치 1회입니다. 웹 경로에서는 자동 API 재시도도
비활성화됩니다.

`.env`에서 다음 값이 필요합니다.

```dotenv
OPENROUTER_API_KEY=your-key
RUN_PAID_EXPERIMENTS=1
WEB_ROUTER_ARTIFACT=artifacts/phase2e/router_anchor_rare_v1.pkl
WEB_HOST=127.0.0.1
WEB_PORT=8000
WEB_CANDIDATE_GATE_ENABLED=true
WEB_DETECTION_MAX_PROMPT_CHARACTERS=120000
WEB_DETECTION_MAX_EXPERT_TASKS=24
WEB_DETECTION_MAX_OUTPUT_TOKENS=8192
WEB_PATCH_MAX_PROMPT_CHARACTERS=120000
```

설치 후 서버를 실행합니다.

```powershell
cd C:\Users\junhyun111\Desktop\llm-security
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
llm-security-web
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. 작업 상태와 업로드 파일은
`.web-data/<job-id>` 아래에 저장됩니다. C/C++ 소스만 정적 분석 및 LLM context로
사용되며 DB, 바이너리, `.git`, build 산출물은 LLM으로 보내지 않습니다. DB를
포함한 나머지 업로드 파일은 승인된 프로젝트 ZIP을 만들기 위해서만 보관합니다.

단일 요청의 컨텍스트와 출력 크기를 통제하기 위해 웹은 기본적으로 최대 24개
Expert 작업과 120,000자의 탐지 프롬프트만 제출합니다. 초과 작업은 추가 요청으로
나누지 않고 결과의 `skipped_expert_task_count`와 `errors`에 기록합니다. 이 값은
`.env`의 `WEB_DETECTION_MAX_EXPERT_TASKS`와
`WEB_DETECTION_MAX_PROMPT_CHARACTERS`로 조절할 수 있습니다. 서로 다른 모델을
Expert마다 실제 실행하는 CLI 실험 경로는 기존대로 유지되지만, 웹의 단일 호출
모드에서는 모든 논리 Expert가 `OPENROUTER_EXPERT_MODEL` 하나를 공유합니다.

Docker로 실행할 수도 있습니다. Router artifact가 `artifacts/phase2e`에 있어야
합니다.

```powershell
docker compose -f compose.web.yml up --build
```

현재 웹 구성은 로컬 단일 사용자 실행을 기본으로 합니다. 인터넷에 공개할 때는
인증, TLS, 외부 작업 큐, 사용자별 저장소 암호화와 보존 기간 정책을 추가해야
합니다. 업로드된 빌드 스크립트나 실행 파일은 자동 실행하지 않으며, 패치 검증은
현재 diff 적용 가능 여부까지만 확인합니다.

## 새 Router 구조와 실행 순서

현재 기본 학습 구조는 single-label Softmax 분류가 아닙니다. `memory_bounds`와
`control_state_error`를 공통 Anchor로 실행하고, 나머지 Expert가 필요한지만 독립
binary trigger로 학습합니다. 실제 LLM 결과를 모은 뒤에는
`P(Expert×Model succeeds | candidate)`를 예측하는 Utility Router로 전환할 수 있습니다.
기존 `AdaptiveExpertRouter`는 비교 baseline으로만 보존됩니다.

1. semantic Candidate JSONL 준비(이미 있으면 생략):

```powershell
python -m llm_security.cli phase2e-prepare `
  --cases data\arvo\cases_all.jsonl `
  --data-dir data\phase2e `
  --backend semantic `
  --seed 2026
```

2. API 없이 Anchor/Rare Router 학습:

```powershell
python -m llm_security.cli train-anchor-router `
  --train data\phase2e\semantic\router_train.jsonl `
  --dev data\phase2e\semantic\router_dev.jsonl `
  --output artifacts\phase2e\router_anchor_rare_v1.pkl
```

같은 작업은 `notebooks/01_train_router.ipynb`를 위에서 아래로 실행해도 됩니다.

3. Utility Router용 Expert×Model 성능 행렬 수집:

`.env`에서 `RUN_PAID_EXPERIMENTS=1`로 바꾸고 먼저 `--max-cases 5`로 호출 수와
응답 형식을 확인합니다. 수집 결과는 매 요청마다 checkpoint되므로 중단 후 같은
명령을 실행하면 이어서 진행합니다.

```powershell
python -m llm_security.cli collect-utility-outcomes `
  --cases data\phase2e\cases_train.jsonl `
  --output data\utility\outcomes_train.jsonl `
  --max-cases 5
```

검증 데이터도 별도로 수집합니다.

```powershell
python -m llm_security.cli collect-utility-outcomes `
  --cases data\phase2e\cases_dev.jsonl `
  --output data\utility\outcomes_dev.jsonl `
  --max-cases 5
```

모델 후보는 `OPENROUTER_SWEEP_MODELS` 또는 `--models model/a,model/b`로 지정합니다.
Expert별 운영 모델은 `OPENROUTER_MEMORY_MODEL`, `OPENROUTER_LIFETIME_MODEL`,
`OPENROUTER_INTEGER_MODEL`, `OPENROUTER_TAINT_MODEL`, `OPENROUTER_CONTROL_MODEL`,
`OPENROUTER_CONCURRENCY_MODEL`로 독립 설정하며, 생략하면
`OPENROUTER_EXPERT_MODEL`을 사용합니다.

4. 수집된 실제 outcome으로 Utility Router 학습:

```powershell
python -m llm_security.cli train-utility-router `
  --train data\utility\outcomes_train.jsonl `
  --dev data\utility\outcomes_dev.jsonl `
  --output artifacts\phase2e\router_utility_v1.pkl
```

`notebooks/02_evaluate_router.ipynb`는 test split의 rare recall, 실제 success
coverage, 평균 Expert 수, 비용과 regret을 평가합니다. `03_run_agents.ipynb`는 Utility
artifact가 있으면 그것을 우선 사용하고, 없으면 Anchor/Rare artifact로 실행합니다.

C/C++ 오픈소스에서 정적 분석으로 취약 후보 함수를 찾고, 학습된 Router가 후보별 Expert를 선택한 뒤 OpenRouter LLM으로 분석하는 실험용 파이프라인입니다.

- 구현 코드: `src/llm_security/*.py`
- Router 학습: `notebooks/01_train_router.ipynb`
- Router 평가: `notebooks/02_evaluate_router.ipynb`
- OpenRouter 에이전트 실행: `notebooks/03_run_agents.ipynb`
- 모델 및 실행 설정: `.env`

## 설치

```powershell
cd C:\Users\junhyun111\Desktop\llm-security
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## ARVO 데이터 준비

`data/arvo/arvo.db`에서 ARVO 레코드를 읽습니다. 취약 위치 정답은 LLM이나 정적 분석 후보에서 만들지 않고 crash stack과 fix patch에서 결정합니다. 취약 소스는 fix commit 파일에 patch를 역적용해 복원합니다.

```powershell
python -m llm_security.cli prepare-arvo `
  --db data\arvo\arvo.db `
  --count 60 `
  --unique-projects `
  --output data\arvo\cases_all.jsonl `
  --dataset-dir data\arvo `
  --seed 2026
```

생성 결과:

- `cases_{train,dev,test}.jsonl`: OpenRouter benchmark 사례
- `router_{train,dev,test}.jsonl`: 실제 ARVO 취약 함수 후보와 Expert 정답
- `manifest.json`: 프로젝트와 family 분포

현재 생성된 60개 사례는 프로젝트 단위로 train 42개, dev 9개, test 9개이며 프로젝트 중복이 없습니다. ARVO fuzzing 레코드에서 확보된 family는 `memory_bounds`, `lifetime_resource`, `control_state_error` 세 종류입니다. 나머지 Expert family를 학습하려면 다른 benchmark를 추가해야 합니다.

공개 GitHub 파일 다운로드가 제한되면 현재 PowerShell 세션에 token을 선택적으로 설정할 수 있습니다.

```powershell
$env:GITHUB_TOKEN="github-token"
```

다운로드 응답은 `data/arvo/cache/github`에 저장되어 재실행 시 재사용됩니다.

### 전체 eligible ARVO 수집

전체 DB 레코드가 아니라, 재현 성공·patch 위치 확인·C/C++·공개 GitHub 저장소 조건을 만족하는 **4,465개**를 대상으로 소스와 patch를 수집하려면 다음을 사용합니다. 같은 프로젝트의 여러 취약점도 포함합니다.

```powershell
python -m llm_security.cli prepare-arvo `
  --all `
  --db data\arvo\arvo.db `
  --output data\arvo\cases_all.jsonl `
  --dataset-dir data\arvo `
  --failure-log data\arvo\arvo_failures.jsonl `
  --seed 2026
```

작업은 25건마다 `cases_all.jsonl`을 checkpoint합니다. `--all`은 기존 `cases_all.jsonl`을 자동으로 재사용하므로 네트워크 제한이나 중단 뒤에는 같은 명령을 그대로 다시 실행하면 됩니다. GitHub에서 삭제·비공개 처리된 저장소, 적용할 수 없는 patch 등은 `arvo_failures.jsonl`에 남고 나머지 사례는 계속 저장됩니다. 전체 수집은 상당한 네트워크 시간과 디스크 공간이 필요합니다.

수집이 중단된 뒤에도 이미 저장된 전체 사례만으로 분할·Router 학습 JSONL을 다시 만들 수 있습니다. 이 명령은 GitHub에 요청하지 않습니다.

```powershell
python -m llm_security.cli split-arvo `
  --cases data\arvo\cases_all.jsonl `
  --dataset-dir data\arvo `
  --seed 2026
```

## Router 학습과 평가

세 노트북은 역할별로 독립되어 있으며 아래 순서로 실행합니다.

```powershell
python -m jupyter lab
```

1. `01_train_router.ipynb`: ARVO train으로 multiclass Softmax Router를 학습하고 `models/router-arvo.pkl` 저장
2. `02_evaluate_router.ipynb`: dev에서 Adaptive Top-k policy를 calibration하고 test에서 평가한 뒤 `models/router-arvo-metrics.json` 저장
3. `03_run_agents.ipynb`: Router와 OpenRouter LLM 에이전트로 실제 benchmark 실행

앞의 두 노트북은 OpenRouter API를 사용하지 않습니다. 세 번째 노트북만 `.env`의 API key와 `RUN_PAID_EXPERIMENTS=1`이 필요합니다. OpenRouter LLM 자체를 fine-tuning하는 것이 아니라 조건부 Expert Router를 학습합니다.

CLI로 학습하려면 다음을 실행합니다. Train은 확률 모델 학습에, dev는 Adaptive Top-k policy calibration에만 사용됩니다.

```powershell
python -m llm_security.cli train-router `
  --train data\arvo\router_train.jsonl `
  --dev data\arvo\router_dev.jsonl `
  --env-file .env `
  --output models\router-arvo.pkl
```

## `.env` 모델 설정

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_EXPERT_MODEL=poolside/laguna-xs-2.1:free
OPENROUTER_VALIDATOR_MODEL=poolside/laguna-xs-2.1:free
OPENROUTER_PATCH_MODEL=poolside/laguna-xs-2.1:free
OPENROUTER_STRONG_MODEL=poolside/laguna-xs-2.1:free
```

전체 변수 설명과 기본값은 `.env.example`에 있습니다. API key는 결과 파일이나 노트북 출력에 기록하지 않습니다.

Adaptive Router 설정은 다음과 같습니다.

```dotenv
CANDIDATE_GATE_THRESHOLD=0.40
ROUTER_HIGH_CONFIDENCE=0.72
ROUTER_MIN_MARGIN=0.18
ROUTER_MAX_ENTROPY=1.0
ROUTER_MAX_EXPERTS=2
ROUTER_TARGET_COVERAGE=0.95
USE_RULE_FALLBACK=true
```

Router 모델은 학습 데이터에 존재하는 Expert family만 class로 학습합니다. 학습되지 않은 Integer·Taint·Concurrency family는 별도 rule-trigger 점수로 보조하며, 결과 JSON에는 learned score와 trigger score가 구분되어 기록됩니다.

## Static analysis frontend

Phase 2A의 Tree-sitter frontend는 C/C++ 소스를 재사용 가능한 IR로 변환합니다.

```text
source files
→ TreeSitterFrontend
→ FunctionIR (statements, controls, calls, assignments, conditions, memory accesses)
→ ProgramIR (1-hop direct callers/callees)
```

Phase 2B의 실험용 `StructuralAnalyzer`는 이 IR만 사용해 함수 단위 CFG, dominator, reaching definitions, def-use graph를 계산합니다. `backward_slice()`로 sink의 변수 정의를 함수 parameter까지 역추적할 수 있습니다.

```text
FunctionIR
→ ControlFlowGraph (true/false/loop_back/return edges)
→ Dominators
→ Reaching Definitions
→ Def-Use Graph
→ Backward Slice
```

구현은 `src/llm_security/analysis/`에 있습니다. 현재 지원하는 구조화 제어 흐름은 `if`, `if/else`, `while`, `for`, 중첩 제어문과 early return입니다. `switch`, `goto`, `break`, `continue`는 임의로 해석하지 않고 CFG warning으로 기록합니다. 기존 regex 기반 analyzer는 `analysis/legacy.py`의 `LegacyRegexAnalyzer`로 보존되며 production pipeline은 계속 기존 `LightweightStaticAnalyzer`를 사용합니다. vulnerability semantic fact, Candidate Gate 연결, Router feature 변경은 Phase 2B 범위에 포함되지 않습니다.

## Semantic fact engine

Phase 2C의 `SemanticAnalyzer`는 구조 분석 결과에 API 역할과 보안 의미를 부여합니다.

```text
ProgramAnalysis
→ ApiCatalog
→ control/data-flow relation
→ SemanticFact + TaintPath
```

기본 catalog는 `malloc/calloc/realloc`, `memcpy/memmove/strncpy/strcpy`, `free`, `read/recv/fread/getenv`, 명령·파일 sink, pthread API와 TOCTOU API를 지원합니다. allocation/release, memory copy와 guard, size arithmetic/cast flow, source-to-sink, use-after-release/double-release, uninitialized use, nullable dereference, lock/thread 및 TOCTOU fact를 생성할 수 있습니다.

`switch`, `goto`, `break`, `continue` 등으로 CFG warning이 있으면 local positive fact는 유지하되 `without_guard`, `unchecked`, `unsanitized` 같은 부재 기반 fact는 생성하지 않습니다. Phase 2C 계층 자체는 Candidate나 Router를 알지 않으며, 아래 Phase 2D adapter를 통해서만 pipeline에 연결됩니다.

## Semantic analysis backend and Candidate Gate

Phase 2D는 semantic fact를 고정된 `semantic-v1` feature, Evidence, suspicion score와 Candidate로 변환합니다. Candidate는 모두 생성한 뒤 Gate에서 별도로 판정하며, `MAX_CANDIDATES` 제한은 Gate 통과 후 적용됩니다.

```text
StructuralAnalyzer → SemanticAnalyzer
→ SemanticFeatureExtractor + SemanticEvidenceNormalizer
→ SuspicionScorer → Candidate
→ CandidateGate → AdaptiveExpertRouter
```

분석 backend는 `.env`에서 선택합니다.

```dotenv
ANALYSIS_BACKEND=legacy
CANDIDATE_GATE_ENABLED=false
CANDIDATE_GATE_THRESHOLD=0.40
```

기본값은 아직 `legacy`입니다. 기존 Router와 semantic Candidate를 섞지 않도록 Candidate와 Router가 `legacy-v1` 또는 `semantic-v1` feature schema를 기록하고, schema가 다르면 inference를 중단합니다. Router artifact format은 v4이며 기존 artifact는 재학습 후 사용해야 합니다. Semantic Router 데이터 재생성과 Gate threshold calibration은 Phase 2E에서 수행합니다.

## 실제 프로젝트 분석

```powershell
python -m llm_security.cli analyze C:\path\to\cpp-project `
  --env-file .env `
  --router-artifact models\router-arvo.pkl `
  --output analysis.json
```

## ARVO test benchmark

```powershell
python -m llm_security.cli run-cases data\arvo\cases_test.jsonl `
  --env-file .env `
  --router-artifact models\router-arvo.pkl `
  --output experiment-arvo.json
```

노트북의 OpenRouter 실행 셀은 `.env`의 `RUN_PAID_EXPERIMENTS=1`일 때만 동작합니다. `:free` 모델도 외부 API 호출 여부를 명시적으로 제어하기 위해 같은 변수를 사용합니다.

## 검증

```powershell
pytest
```

## Phase 2E offline experiment

ARVO case 전체를 고정된 project-disjoint split으로 나누고 Legacy/Semantic
analyzer, Candidate Gate, Softmax Router, Adaptive Top-K 및 rule fallback을 같은
조건에서 비교합니다. 이 명령은 OpenRouter 또는 다른 LLM API를 호출하지 않습니다.
CLI는 `cases_all.jsonl`을 한 건씩 읽는 스트리밍 경로를 사용하므로 전체 소스
코퍼스를 메모리에 동시에 올리지 않습니다. Router JSONL에는 학습에 필요한
feature와 label만 남기고 원본 함수 코드와 Evidence는 제외합니다.

```powershell
python -m llm_security.cli phase2e `
  --cases data\arvo\cases_all.jsonl `
  --data-dir data\phase2e `
  --artifacts-dir artifacts\phase2e `
  --output results\phase2e `
  --seed 2026
```

분할과 backend별 Router JSONL은 `data/phase2e`, 학습된 Router는
`artifacts/phase2e`, 평가 결과는 `results/phase2e`에 저장됩니다.

전처리와 notebook 학습을 분리하려면 먼저 아래 명령만 실행합니다.

```powershell
python -m llm_security.cli phase2e-prepare `
  --cases data\arvo\cases_all.jsonl `
  --data-dir data\phase2e `
  --seed 2026
```

Semantic preprocessing has a 2 MiB per-source-file safety cap and a 30-second
Tree-sitter parse timeout by default. Oversized or timed-out cases are retained
in the benchmark denominator and recorded in
`data/phase2e/analysis_failures/semantic_<split>.jsonl`. Override them with
`--max-source-mb` and `--parse-timeout-seconds` when needed.

그 다음 `notebooks/01_train_router.ipynb`의 셀을 위에서부터 실행하면
`artifacts/phase2e/router_legacy_v1.pkl`과
`artifacts/phase2e/router_semantic_v1.pkl`이 생성됩니다.

함수 경계는 Tree-sitter C/C++ AST로 추출합니다. 현재 Router 특징은 portable 정적 신호이며, 이후 Clang·CodeQL·Joern adapter가 동일한 `Candidate`와 `Evidence` schema를 출력하도록 확장할 수 있습니다.
