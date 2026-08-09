# LLM Security Conditional Expert Pipeline

C/C++ 오픈소스에서 정적 분석으로 취약 후보 함수를 찾고, 학습된 Router가 후보별 Expert를 선택한 뒤 OpenRouter LLM으로 분석하는 실험용 파이프라인입니다.

- 구현 코드: `src/llm_security/*.py`
- ARVO 학습·평가 노트북: `notebooks/01_model_experiments.ipynb`
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
  --output data\arvo\processed\cases_all.jsonl `
  --dataset-dir data\arvo\processed `
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

## Router 학습과 평가

```powershell
python -m jupyter lab notebooks\01_model_experiments.ipynb
```

노트북은 합성 데이터를 사용하지 않습니다. ARVO train만 학습하고 dev와 test를 따로 평가한 뒤 `models/router-arvo.pkl`을 저장합니다. OpenRouter LLM 자체를 fine-tuning하는 것이 아니라 조건부 Expert Router를 학습합니다.

CLI로 학습하려면 다음을 실행합니다. 최종 성능 확인 전에는 `--test`에 dev 파일을 사용하는 것을 권장합니다.

```powershell
python -m llm_security.cli train-router `
  --train data\arvo\processed\router_train.jsonl `
  --test data\arvo\processed\router_dev.jsonl `
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

## 실제 프로젝트 분석

```powershell
python -m llm_security.cli analyze C:\path\to\cpp-project `
  --env-file .env `
  --router-artifact models\router-arvo.pkl `
  --output analysis.json
```

## ARVO test benchmark

```powershell
python -m llm_security.cli run-cases data\arvo\processed\cases_test.jsonl `
  --env-file .env `
  --router-artifact models\router-arvo.pkl `
  --output experiment-arvo.json
```

노트북의 OpenRouter 실행 셀은 `.env`의 `RUN_PAID_EXPERIMENTS=1`일 때만 동작합니다. `:free` 모델도 외부 API 호출 여부를 명시적으로 제어하기 위해 같은 변수를 사용합니다.

## 검증

```powershell
pytest
```

함수 경계는 Tree-sitter C/C++ AST로 추출합니다. 현재 Router 특징은 portable 정적 신호이며, 이후 Clang·CodeQL·Joern adapter가 동일한 `Candidate`와 `Evidence` schema를 출력하도록 확장할 수 있습니다.
