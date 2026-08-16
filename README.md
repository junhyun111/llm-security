# LLM Security

C/C++ 프로젝트를 업로드하면 취약점을 분석하고, 사용자가 승인한 항목만 수정하는 웹 애플리케이션입니다. 학습된 Router가 저장소에 포함되어 있으므로 추가 학습 없이 실행할 수 있습니다.

## 실행 방법

### 1. 설치

Python 3.13 환경에서 다음 명령을 실행합니다.

```powershell
git clone https://github.com/junhyun111/llm-security.git
cd llm-security

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### 2. OpenRouter 설정

`.env.example`을 `.env`로 복사합니다.

```powershell
Copy-Item .env.example .env
notepad .env
```

최소한 다음 값을 설정합니다.

```dotenv
OPENROUTER_API_KEY=발급받은_API_KEY
OPENROUTER_EXPERT_MODEL=google/gemini-3.7-flash
OPENROUTER_PATCH_MODEL=google/gemini-3.7-flash
RUN_PAID_EXPERIMENTS=1
```

### 3. 웹 서버 실행

```powershell
python -m llm_security.web.run
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8000
```

웹에서 C/C++ 프로젝트 폴더를 선택하면 분석이 시작됩니다. 취약점 결과를 확인한 뒤 수정할 항목을 선택하면 통합 patch가 생성되고, 승인 후 수정된 프로젝트를 내려받을 수 있습니다.

Docker를 사용하는 경우 `.env` 설정 후 다음 명령으로 실행할 수 있습니다.

```powershell
docker compose -f compose.web.yml up --build
```

## 모델 구조

```text
C/C++ 프로젝트 업로드
        ↓
정적 분석 (AST · CFG · Data Flow · CWE 가설)
        ↓
Candidate Gate
        ↓
Anchor/Rare Router
  ├─ E1 Memory Safety
  ├─ E2 Lifetime / Resource
  ├─ E3 Integer / Size / Type
  ├─ E4 Taint / API Contract
  ├─ E5 Control / State / Error
  └─ E6 Concurrency / TOCTOU
        ↓
선택된 Expert를 하나의 OpenRouter 요청으로 통합 실행
        ↓
취약점 검증 · 중복 제거 · 한국어 결과 출력
        ↓
사용자 승인
        ↓
통합 patch 생성 · 적용 검증 · 수정 프로젝트 다운로드
```

Router는 `artifacts/phase2e/router_anchor_rare_v2.pkl`을 사용합니다. E1과 E5는 기본 Anchor로 실행하고, 나머지 Expert는 코드 특징과 정적 CWE 근거 및 학습된 임계값에 따라 선택합니다. 취약점 탐지는 프로젝트당 통합 API 요청 1회, patch 생성은 사용자 승인 후 1회를 기본으로 합니다.
