# Utility Router Runtime

이 폴더는 학습과 평가 코드를 제외한 독립 배포 폴더입니다. 고정된 Candidate
Ranker와 추론 코드가 `src` 아래에 함께 들어 있으므로, 폴더를 별도로 복사한 뒤
`artifacts/router.pkl`만 교체하여 사용할 수 있습니다.

## 포함 파일

- `artifacts/router.pkl`: 교체 가능한 학습 Utility Router
- `artifacts/candidate_ranker.pkl`: 배포에 고정된 Candidate Ranker
- `run.sh`, `run.cmd`: Bash/Windows 실행 진입점
- `src/llm_security_runtime`: artifact 검증, CLI, FastAPI 연결 코드
- `src/llm_security`, `src/model_evaluation`: 학습 코드를 제외한 추론 코드

`router.pkl`은 서버가 관리하는 신뢰된 파일만 사용해야 합니다. Python pickle은
임의 코드를 실행할 수 있으므로 API 사용자가 업로드한 모델을 직접 로드하면 안
됩니다.

## 1. 설정

Bash:

```bash
cd model_runtime
cp .env.example .env
bash setup.sh
```

Windows CMD:

```cmd
cd model_runtime
copy .env.example .env
setup.cmd
```

`.env`에 `OPENROUTER_API_KEY`를 입력하고 실제 분석 전에
`RUN_PAID_EXPERIMENTS=1`로 변경합니다.

## 2. artifact 확인

OpenRouter 요청 없이 두 모델의 버전, schema, SHA-256 및 모델 호환성을
확인합니다.

```bash
bash run.sh inspect
```

```cmd
run.cmd inspect
```

## 3. CLI 분석

```bash
bash run.sh analyze ../target-project --output analysis.json
```

```cmd
run.cmd analyze ..\target-project --output analysis.json
```

## 4. 백엔드 서버

```bash
bash run.sh serve --host 0.0.0.0 --port 8000
```

서버 확인:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/runtime
```

프로젝트 업로드:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "project_name=sample" \
  -F "relative_paths=main.c" \
  -F "files=@main.c"
```

응답의 `job_id`로 상태와 결과를 조회합니다.

```bash
curl http://127.0.0.1:8000/api/jobs/JOB_ID
curl http://127.0.0.1:8000/api/jobs/JOB_ID/analysis
```

## Router 교체 계약

새 학습 파일을 `artifacts/router.pkl`이라는 이름으로 교체한 뒤 `inspect`를
실행합니다. 현재 런타임은 다음 계약을 검사합니다.

- `BudgetedUtilityRouter` artifact version 5
- `semantic-cwe-v3` feature schema
- 다섯 Utility Expert assignment
- Candidate Ranker와 동일한 feature schema
- `.env`의 `OPENROUTER_EXPERT_MODEL`과 Router 학습 model ID 일치

실행 중인 작업과 파일 교체가 겹치지 않도록 서버를 중지한 상태에서 교체하는
것을 권장합니다.
