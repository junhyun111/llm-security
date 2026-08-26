# LLM Security Web Platform

C/C++ 프로젝트를 업로드하면 정적 분석과 LLM 기반 Expert 검증을 통해 보안 취약점을 탐지하고, 사용자가 승인한 항목에 대해서만 통합 패치를 생성·적용하는 웹 애플리케이션입니다.

`feature/web-platform` 브랜치에서는 기존 Python 분석 엔진에 다음 기능을 추가했습니다.

- ASP.NET Core **.NET 10** 백엔드
- **SQLite + Entity Framework Core** 기반 데이터 저장
- ASP.NET Core Identity 기반 **회원가입 / 로그인 / 로그아웃**
- 사용자별 **분석 이력 / 취약점 / 패치 기록 관리**
- **React + TypeScript + Vite** 기반 신규 프론트엔드
- 기존 Python 분석 엔진과 .NET 백엔드 연동

---

## 1. 전체 구조

```text
사용자
  │
  ▼
React Frontend
http://localhost:5173
  │
  ▼
ASP.NET Core .NET 10
http://localhost:5080
  │
  ├──────────────► SQLite
  │                사용자 / 분석 이력 / 취약점 / 패치
  │
  ▼
Python FastAPI Analyzer
http://127.0.0.1:8000
  │
  ▼
정적 분석 + Router + OpenRouter Expert
```

기존 Python 분석 엔진은 그대로 유지하며, .NET 백엔드가 Python API를 호출하는 방식으로 구성되어 있습니다.

---

## 2. 주요 기능

- 회원가입 / 로그인 / 로그아웃
- 사용자별 분석 기록 분리
- C/C++ 프로젝트 폴더 업로드
- 분석 진행 상태 확인
- 취약점 결과 조회
- 검증된 취약점 선택
- 통합 패치 생성
- 패치 승인 / 거절
- 수정된 프로젝트 ZIP 다운로드
- 전체 분석 / 완료 분석 / 취약점 / 승인 패치 대시보드

---

## 3. 분석 파이프라인

```text
C/C++ 프로젝트 업로드
        ↓
정적 분석 (AST · CFG · Data Flow · CWE 가설)
        ↓
Candidate Gate
        ↓
Anchor / Rare Router
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
사용자 선택
        ↓
통합 Patch 생성
        ↓
승인 / 거절
        ↓
수정 프로젝트 다운로드
```

Router는 다음 학습된 모델을 사용합니다.

```text
artifacts/phase2e/router_anchor_rare_v2.pkl
```

---

# 실행 방법

## 4. 요구 환경

Windows PowerShell 기준입니다.

| 프로그램 | 권장 버전 | 용도 |
|---|---:|---|
| Git | 최신 | 저장소 Clone |
| Python | 3.13 | 기존 분석 엔진 |
| .NET SDK | 10.x | 백엔드 |
| Node.js | LTS | React 프론트엔드 |
| npm | Node.js 설치 시 포함 | 프론트 패키지 |
| OpenRouter API Key | - | LLM Expert 분석 |

> SQLite는 별도 설치할 필요가 없습니다. .NET 백엔드 실행 시 DB 파일이 자동 생성됩니다.

설치 확인:

```powershell
git --version
python --version
dotnet --version
node --version
npm --version
```

---

## 5. 저장소 Clone

이 브랜치를 바로 Clone하는 경우:

```powershell
git clone -b feature/web-platform https://github.com/junhyun111/llm-security.git
cd llm-security
```

이미 저장소가 있는 경우:

```powershell
git fetch origin
git switch feature/web-platform
git pull
```

---

## 6. Python 환경 설정

프로젝트 루트에서:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

---

## 7. OpenRouter 설정

`.env.example`을 `.env`로 복사합니다.

```powershell
Copy-Item .env.example .env
notepad .env
```

최소 설정:

```dotenv
OPENROUTER_API_KEY=발급받은_실제_API_KEY
OPENROUTER_EXPERT_MODEL=google/gemini-3.7-flash
OPENROUTER_PATCH_MODEL=google/gemini-3.7-flash
RUN_PAID_EXPERIMENTS=1
```

> `.env`와 API Key는 GitHub에 Commit하지 마세요.

---

# 서버 실행

전체 웹 플랫폼을 사용하려면 **PowerShell 터미널 3개**를 실행합니다.

## 8. 터미널 1 - Python 분석 서버

프로젝트 루트:

```powershell
.\.venv\Scripts\Activate.ps1
python -m llm_security.web.run
```

주소:

```text
http://127.0.0.1:8000
```

---

## 9. 터미널 2 - .NET 백엔드

```powershell
cd backend\LlmSecurity.Api
dotnet restore
dotnet run
```

정상 실행:

```text
Now listening on: http://localhost:5080
```

Health Check:

```text
http://localhost:5080/api/health
```

Python 서버까지 연결되어 있으면:

```json
{
  "status": "ok",
  "analyzer": "ok"
}
```

---

## 10. 터미널 3 - React 프론트엔드

```powershell
cd frontend
npm install
npm run dev
```

정상 실행:

```text
VITE ready
Local: http://localhost:5173/
```

접속:

```text
http://localhost:5173
```

---

## 11. 첫 실행 / 회원가입

예시:

```text
이름: test
이메일: test@test.com
비밀번호: test1234
```

현재 비밀번호 기본 조건:

- 8자 이상
- 영문 소문자 포함
- 숫자 포함

회원가입 성공 시 자동 로그인됩니다.

---

## 12. SQLite DB

.NET 백엔드 최초 실행 시 DB가 자동 생성됩니다.

```text
backend/LlmSecurity.Api/data/llm-security.db
```

주요 구조:

```text
AspNetUsers
    │
    │ 1:N
    ▼
AnalysisJobs
    │
    ├── 1:N ─── AnalysisFindings
    │
    └── 1:0..1 ─ PatchBatches
```

주요 테이블:

- `AspNetUsers`: 사용자 계정
- `AnalysisJobs`: 분석 프로젝트 및 상태
- `AnalysisFindings`: 취약점 결과
- `PatchBatches`: 통합 패치 기록

---

## 13. 사용 흐름

로그인 후:

```text
New Scan
  ↓
프로젝트 폴더 선택
  ↓
보안 분석 시작
```

전체 요청 흐름:

```text
React
  ↓
.NET Backend
  ↓
Python Analyzer
  ↓
OpenRouter
  ↓
Python 분석 결과
  ↓
.NET Backend
  ↓
SQLite 저장
  ↓
React 결과 화면
```

검증된 취약점은 사용자가 선택하여 통합 패치에 포함할 수 있습니다.

---

# API 구조

## 14. .NET Backend API

```text
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me

GET  /api/dashboard

GET  /api/analyses
POST /api/analyses
GET  /api/analyses/{id}

POST /api/analyses/{id}/patches/proposal
POST /api/analyses/{id}/patches/{patchId}/approve
POST /api/analyses/{id}/patches/{patchId}/reject

GET /api/analyses/{id}/download
```

## 15. 기존 Python Analyzer API

.NET 백엔드는 기존 Python API를 내부적으로 호출합니다.

```text
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/analysis

POST /api/jobs/{job_id}/patches/proposal
POST /api/jobs/{job_id}/patches/{patch_id}/approve
POST /api/jobs/{job_id}/patches/{patch_id}/reject

GET /api/jobs/{job_id}/download
```

Python 분석 알고리즘을 C#으로 재작성하지 않습니다.

---

# 프로젝트 구조

```text
llm-security/
│
├── src/
│   └── llm_security/            # 기존 Python 분석 엔진
│
├── backend/
│   └── LlmSecurity.Api/
│       ├── Controllers/
│       ├── Data/
│       ├── DTOs/
│       ├── Models/
│       ├── Services/
│       ├── Program.cs
│       └── appsettings.json
│
├── frontend/
│   ├── src/
│   │   ├── auth/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── styles.css
│   │   └── types.ts
│   └── package.json
│
├── artifacts/
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

---

# 문제 해결

## 16. `node` / `npm`을 찾을 수 없음

Node.js LTS 설치 후 PowerShell을 완전히 다시 실행합니다.

```powershell
node --version
npm --version
```

---

## 17. Python 분석 서버에 연결할 수 없음

웹에 다음 메시지가 뜨는 경우:

```text
Python 분석 서버에 연결할 수 없습니다.
분석 서버가 실행 중인지 확인해주세요.
```

Python Analyzer를 실행합니다.

```powershell
python -m llm_security.web.run
```

그리고 다음 주소를 확인합니다.

```text
http://localhost:5080/api/health
```

`analyzer`가 `ok`여야 합니다.

---

## 18. OpenRouter 응답 길이 초과

다음 오류:

```text
Model returned incomplete or invalid JSON
finish_reason=length
```

은 프론트엔드 / .NET / SQLite 문제가 아니라 OpenRouter 모델의 출력이 토큰 제한에 도달해 JSON 생성이 중간에 종료된 경우입니다.

테스트 시 `.env`에 Expert 작업 수 제한을 추가할 수 있습니다.

```dotenv
WEB_DETECTION_MAX_EXPERT_TASKS=2
```

변경 후 Python 서버를 재시작합니다.

```powershell
Ctrl + C
python -m llm_security.web.run
```

---

## 19. SQLite NuGet 취약점 경고

확인:

```powershell
dotnet list package --vulnerable --include-transitive
```

취약한 transitive package가 있으면 안전한 최신 패치 버전으로 업데이트합니다.

---

# Git에 올리면 안 되는 항목

```text
.env
.venv/
frontend/node_modules/
frontend/dist/
backend/LlmSecurity.Api/data/
data/backend/
```

특히 `OPENROUTER_API_KEY`는 절대 Commit하지 않습니다.

---

# 빠른 실행 요약

### Terminal 1

```powershell
cd llm-security
.\.venv\Scripts\Activate.ps1
python -m llm_security.web.run
```

### Terminal 2

```powershell
cd llm-security\backend\LlmSecurity.Api
dotnet run
```

### Terminal 3

```powershell
cd llm-security\frontend
npm run dev
```

접속:

```text
http://localhost:5173
```

---

# 기술 스택

```text
Frontend
- React
- TypeScript
- Vite

Backend
- ASP.NET Core
- .NET 10
- Entity Framework Core
- ASP.NET Core Identity

Database
- SQLite

Analysis
- Python 3.13
- AST / CFG / Data Flow
- Anchor / Rare Router
- OpenRouter LLM Experts
```
