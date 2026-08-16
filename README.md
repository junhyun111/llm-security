# LLM Security Conditional Expert Pipeline

## ARVO + Juliet Router 학습 빠른 실행

`notebooks/01_train_router.ipynb`를 열고 셀을 위에서부터 실행하면 됩니다. 기본 설정은
다음 Juliet 경로를 자동으로 사용합니다.

```text
C:\Users\junhyun111\Downloads\2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3
```

경로가 다르면 첫 번째 코드 셀의 `JULIET_SOURCE`를 바꾸거나 Jupyter를 실행하기 전에
`JULIET_SOURCE_DIR` 환경 변수를 설정합니다. 노트북은 다음 작업을 자동으로 수행합니다.

1. Juliet에서 E3 Integer, E4 Taint/API, E6 Concurrency 사례를 family/CWE별로 균형 추출
2. 번호만 다른 flow variant를 같은 template project로 묶어 train/dev/test 누수 방지
3. 100 case마다 체크포인트를 저장하며 Juliet만 semantic 정적 분석
4. 기존 ARVO Router JSONL과 Juliet Router JSONL을 `data/phase2e_combined`에 병합
5. 합친 표본으로 Anchor/Rare Router 학습 및 dev threshold 보정

중단 후 노트북을 다시 실행하면 정적 분석 체크포인트에서 이어집니다. 처음부터 다시 만들
때만 설정 셀의 `REBUILD_JULIET`, `REBUILD_JULIET_FEATURES` 중 필요한 값을 `True`로
바꾸십시오. 기존 `data/phase2e/semantic` ARVO 분석 결과는 재사용하므로 ARVO 4,160개를
다시 분석하지 않습니다. 학습 모델은
`artifacts/phase2e/router_anchor_rare_v2.pkl`에 저장됩니다. 이 과정에는 OpenRouter API key가
필요하지 않습니다. 실제 Expert 실행 결과가 `data/utility/outcomes_train.jsonl`과
`outcomes_dev.jsonl`에 있을 때만 Utility Router도 추가로 학습합니다.

CLI로 Juliet feature 전처리만 수행하려면 고정 분할을 명시합니다. 그 뒤 compact Router
JSONL 병합과 학습은 노트북이 수행합니다.

```powershell
python -m llm_security.cli phase2e-prepare `
  --frozen-splits-dir data\juliet `
  --data-dir data\phase2e_juliet `
  --backend semantic `
  --seed 2026
```

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
WEB_ROUTER_ARTIFACT=artifacts/phase2e/router_top2_full5_v4.pkl
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

최종 Router는 `Adaptive Top-2 / Full-5 Escalation` 정책입니다. E1 Memory Safety는
기존 bounds와 lifetime 검사를 합치며, E3 Integer/Size/Type, E4 Taint/API Contract,
E5 Control/State/Error, E6 Concurrency/TOCTOU와 함께 5개 Expert를 사용합니다. 각
`Expert×Model`의 성공 확률을 독립적으로 학습하고, 오탐·근거 없는 주장·실측 비용을
뺀 Utility로 Expert별 최적 모델을 하나씩 고른 뒤 순위를 정합니다. 정상 경로는
Top-2이고, 학습된 Escalation Gate의 Top2Sufficient 확률이 보정 threshold보다 낮으면
Full-5로 확장합니다. 기존 AnchorRare는 baseline과 새 artifact 부재 시 fallback입니다.

1. semantic Candidate JSONL 준비(이미 있으면 생략):

```powershell
python -m llm_security.cli phase2e-prepare `
  --cases data\arvo\cases_all.jsonl `
  --data-dir data\phase2e `
  --backend semantic `
  --seed 2026
```

2. 선택 사항: API 없이 Anchor/Rare baseline 학습:

```powershell
python -m llm_security.cli train-anchor-router `
  --train data\phase2e\semantic\router_train.jsonl `
  --dev data\phase2e\semantic\router_dev.jsonl `
  --output artifacts\phase2e\router_anchor_rare_v2.pkl
```

3. Utility Router용 Expert×Model 성능 행렬 수집:

`.env`에서 `RUN_PAID_EXPERIMENTS=1`로 바꾸고 먼저 `--max-cases 5`로 호출 수와
응답 형식을 확인합니다. 수집 결과는 매 요청마다 checkpoint되므로 중단 후 같은
명령을 실행하면 이어서 진행합니다.

Outcome 정답은 단순 line overlap이 아니라 `LocationMatch AND EvidenceValid AND
SemanticCompatible`로 판정됩니다. 현재 label version은 `semantic-causal-v1`입니다. 이전
형식의 outcome 파일은 학습과 resume에서 거부되므로, 기존 파일을 보관한 뒤
`--no-resume`으로 다시 수집해야 합니다. GT 규칙 변경만으로 API를 자동 호출하지는
않습니다.

```powershell
python -m llm_security.cli collect-utility-outcomes `
  --cases data\benchmark\cases_train.jsonl `
  --output data\utility\outcomes_train.jsonl `
  --max-cases 5
```

검증과 최종 test 데이터도 프로젝트가 겹치지 않게 별도로 수집합니다.

```powershell
python -m llm_security.cli collect-utility-outcomes `
  --cases data\benchmark\cases_dev.jsonl `
  --output data\utility\outcomes_dev.jsonl `
  --max-cases 5
```

```powershell
python -m llm_security.cli collect-utility-outcomes `
  --cases data\benchmark\cases_test.jsonl `
  --output data\utility\outcomes_test.jsonl `
  --max-cases 5
```

모델 후보는 `OPENROUTER_SWEEP_MODELS` 또는 `--models model/a,model/b`로 지정합니다.
Expert별 운영 모델은 `OPENROUTER_MEMORY_MODEL`, `OPENROUTER_INTEGER_MODEL`,
`OPENROUTER_TAINT_MODEL`, `OPENROUTER_CONTROL_MODEL`,
`OPENROUTER_CONCURRENCY_MODEL`로 독립 설정하며, 생략하면
`OPENROUTER_EXPERT_MODEL`을 사용합니다. `OPENROUTER_LIFETIME_MODEL`은 기존
AnchorRare 실행 호환용입니다.

4. 수집된 실제 outcome으로 Utility Router 학습:

```powershell
python -m llm_security.cli train-utility-router `
  --train data\utility\outcomes_train.jsonl `
  --dev data\utility\outcomes_dev.jsonl `
  --target-truth-recall 0.95 `
  --output artifacts\phase2e\router_top2_full5_v4.pkl
```

`--gate-train`을 생략하면 dev 프로젝트를 절반씩 나눠 한쪽으로 Escalation Gate를
학습하고 다른 쪽으로 recall 제약하에서 최소 비용 threshold를 고릅니다. test는 이
명령에서 읽지 않습니다. Best Single Expert/Model과 Best Fixed-2도 이 calibration
split에서 고정하며 test 결과를 보고 선택하지 않습니다. CWE hypothesis 입력을 포함한
artifact format은 v4이므로 기존 파일이 있더라도 위 명령으로 다시 학습해야 합니다. 학습 완료 후
test는 한 번만 평가합니다.

```powershell
python -m llm_security.cli evaluate-utility-router `
  --artifact artifacts\phase2e\router_top2_full5_v4.pkl `
  --anchor-artifact artifacts\phase2e\router_anchor_rare_v2.pkl `
  --test data\utility\outcomes_test.jsonl `
  --output results\utility_test.json
```

평가 보고서는 Best Single, Best Fixed-2, Full-5, 고정 E1+E3, Utility Top-2,
독립확률 escalation, 학습 gate를 같이 비교합니다. truth recall, precision/F1, 평균
Expert 수, Full-5 비율뿐 아니라 Escalation Recall, Missed/Unnecessary Escalation Rate,
Expert predictor와 Gate 각각의 Brier/ECE를 기록합니다. 요청량은
`logical_expert_tasks`, 개별 모델 outcome 수집의 `research_physical_requests`, 웹에서
프로젝트 단위로 묶인 `web_batched_requests`로 분리합니다.

정책 비교 표와 `Recall vs Average Experts`, `Recall vs API Cost` 그래프는 평가 JSON에서
생성합니다.

```powershell
python -m llm_security.cli plot-utility-results `
  --input results\utility_test.json `
  --output-dir results\utility_figures
```

### 전체 파이프라인 오프라인 평가

Router outcome에 포함된 후보만 평가하면 정적 분석기와 Candidate Gate에서 놓친 GT가
보이지 않습니다. 아래 명령은 test case를 다시 정적 분석하고 Gate → Router → 이미
수집한 post-validator outcome을 재생합니다. OpenRouter는 호출하지 않습니다.

```powershell
python -m llm_security.cli evaluate-utility-end-to-end `
  --cases data\benchmark\cases_test.jsonl `
  --outcomes data\utility\outcomes_test.jsonl `
  --artifact artifacts\phase2e\router_top2_full5_v4.pkl `
  --candidate-gate-enabled `
  --output results\utility_end_to_end.json
```

결과에는 Analyzer Candidate Recall, Candidate Gate GT Retention, outcome matrix GT
coverage, Full-5 oracle recall, routed detection recall, post-validator precision과 최종
End-to-End F1이 따로 기록됩니다. `outcome_matrix_gt_coverage`가 낮으면 Router가 아니라
outcome 수집 범위부터 보완해야 합니다.

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

현재 로컬 ARVO JSONL은 4,160개 사례이며 프로젝트 단위로 train 2,940개, dev
485개, test 735개로 분리되어 프로젝트 중복이 없습니다. ARVO에서 확보된 family는
주로 `memory_bounds`, `lifetime_resource`, `control_state_error`이므로 E3/E4/E6를
검증하려면 아래 benchmark를 추가해야 합니다.

### E3/E4/E6용 NIST Juliet 추가

ARVO만으로 부족한 E3 Integer, E4 Taint/API, E6 Concurrency 양성 사례는 압축을 푼
Juliet C/C++ 소스 폴더에서 변환합니다. 변환기는 `bad` 계열 함수와 `FLAW` 표시 위치만
GT로 사용하고, 번호만 다른 Juliet flow variant를 같은 template project로 묶어
train/dev/test 간 near-clone 누수를 막습니다. 이 단계는 네트워크나 LLM API를 사용하지
않습니다.

```powershell
python -m llm_security.cli prepare-juliet `
  --source C:\datasets\juliet\C\testcases `
  --output-dir data\juliet `
  --seed 2026
```

ARVO와 Juliet의 이미 고정된 split은 다음처럼 합칩니다. 같은 `project_id`가 서로 다른
split에 있거나 `case_id`가 중복되면 병합을 중단합니다.

```powershell
python -m llm_security.cli merge-case-splits `
  --inputs data\arvo data\juliet `
  --output-dir data\benchmark
```

생성된 `data/benchmark/cases_{train,dev,test}.jsonl`에서 각 Expert가 train/dev/test에
모두 존재하는지 `split_manifest.json`의 `family_distribution`으로 먼저 확인합니다.

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

Phase 2D는 semantic fact를 고정된 `semantic-cwe-v2` feature, Evidence, suspicion
score, 정적 CWE hypothesis와 Candidate로 변환합니다. CWE hypothesis는 GT가 아니라
코드와 Evidence만으로 만든 반증 가능한 후보입니다. Candidate는 모두 생성한 뒤
Gate에서 별도로 판정하며, `MAX_CANDIDATES` 제한은 Gate 통과 후 적용됩니다.

CWE 포함 경로는 다음처럼 동작합니다.

```text
코드 -> Semantic Evidence -> 정적 CWE 가설과 신뢰도
     -> CWE별/계열별 Router feature -> Top-2 또는 Full-5 Expert
     -> LLM의 CWE 확인·기각·수정 -> Evidence 기반 Validator
```

정적 CWE 가설은 ARVO나 Juliet의 정답 CWE를 읽지 않습니다. 학습 데이터의 정답은
Expert utility label을 만드는 데만 사용되고, 실제 추론 시 Router 입력은 업로드된 코드와
정적 분석 Evidence로만 계산됩니다. 따라서 CWE 이름만 맞추는 분류기가 아니라
`어떤 Expert를 호출해야 검증된 취약점을 가장 잘 찾는가`를 학습합니다.

```text
StructuralAnalyzer → SemanticAnalyzer
→ SemanticFeatureExtractor + SemanticEvidenceNormalizer
→ SuspicionScorer → Candidate
→ CandidateGate → AdaptiveExpertRouter
```

분석 backend는 `.env`에서 선택합니다.

```dotenv
ANALYSIS_BACKEND=semantic
CANDIDATE_GATE_ENABLED=false
CANDIDATE_GATE_THRESHOLD=0.40
```

기본값은 `semantic`입니다. 기존 Router와 semantic Candidate를 섞지 않도록
Candidate와 Router가 `legacy-v1` 또는 `semantic-cwe-v2` feature schema를 기록하고,
schema가 다르면 inference를 중단합니다. Utility Router artifact format은 v4이며 기존
artifact와 기존 semantic Router JSONL은 재사용하지 않습니다. Semantic Router 데이터
재생성과 Gate threshold calibration은 Phase 2E에서 수행합니다.

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
  --backend semantic `
  --seed 2026
```

Semantic preprocessing has a 2 MiB per-source-file safety cap and a 30-second
Tree-sitter parse timeout by default. Oversized or timed-out cases are retained
in the benchmark denominator and recorded in
`data/phase2e/analysis_failures/semantic_<split>.jsonl`. Override them with
`--max-source-mb` and `--parse-timeout-seconds` when needed.

Semantic 분석은 기본적으로 100 case마다
`data/phase2e/analysis_checkpoints/semantic_<split>_prepare.json`에 원자적으로
checkpoint를 저장합니다. `Ctrl+C`로 중단해도 같은 명령을 다시 실행하면 마지막
checkpoint 다음 case부터 자동 재개합니다. 중단 시점과 가장 최근 checkpoint 사이의
최대 99 case만 다시 분석합니다. 처음부터 다시 실행하려면 `--no-resume`을 추가하고,
간격을 바꾸려면 `--checkpoint-every 50`처럼 지정합니다.

그 다음 `notebooks/01_train_router.ipynb`의 셀을 위에서부터 실행하면
`artifacts/phase2e/router_anchor_rare_v2.pkl`이 생성됩니다. 현재 prompt/schema로
수집한 `data/utility/outcomes_train.jsonl`과 `outcomes_dev.jsonl`도 있으면
`artifacts/phase2e/router_top2_full5_v4.pkl`까지 생성됩니다. 기존
`semantic-v1` JSONL과 v1-v3 Router artifact는 CWE feature가 없으므로 재사용할 수
없습니다. 먼저 `phase2e-prepare --backend semantic`으로 Router JSONL을 다시 만든 뒤,
현재 prompt/schema로 utility outcome을 다시 수집해 학습해야 합니다.

함수 경계는 Tree-sitter C/C++ AST로 추출합니다. 현재 Router 특징은 portable 정적 신호이며, 이후 Clang·CodeQL·Joern adapter가 동일한 `Candidate`와 `Evidence` schema를 출력하도록 확장할 수 있습니다.
