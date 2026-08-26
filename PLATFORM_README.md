# LLM Security Web Platform 추가본

이 폴더들은 기존 Python 분석 엔진을 대체하지 않습니다.

구조:

```text
브라우저 (React)
      ↓
ASP.NET Core 10 API
      ├─ SQLite / Identity
      └─ HTTP
          ↓
기존 Python FastAPI 분석기
```

## 0. 반드시 확인

저장소 루트에서 현재 브랜치가 아래처럼 보여야 합니다.

```powershell
git branch
# * feature/web-platform
#   main
```

## 1. 이 압축 파일 넣기

압축 내부의 다음 항목을 `llm-security` 저장소 루트에 그대로 복사합니다.

```text
backend/
frontend/
compose.platform.yml
PLATFORM_README.md
```

기존 아래 파일은 건드리지 않습니다.

```text
src/
artifacts/
Dockerfile.web
compose.web.yml
pyproject.toml
```

## 2. 가장 쉬운 실행 방법: Docker

기존 `.env`에 OpenRouter 설정이 되어 있어야 합니다.

최소:

```env
OPENROUTER_API_KEY=...
OPENROUTER_EXPERT_MODEL=google/gemini-3.7-flash
OPENROUTER_PATCH_MODEL=google/gemini-3.7-flash
RUN_PAID_EXPERIMENTS=1
```

저장소 루트 PowerShell:

```powershell
docker compose -f compose.platform.yml up --build
```

빌드가 끝나면:

- 새 웹사이트: http://localhost:3000
- .NET API: http://localhost:5080
- 기존 Python 분석기: http://localhost:8000

처음 회원가입하면 SQLite 파일이 자동 생성됩니다.

```text
data/backend/llm-security.db
```

Docker 중지:

```powershell
docker compose -f compose.platform.yml down
```

DB와 분석 기록까지 초기화하려면 컨테이너를 내린 뒤 아래 폴더를 지웁니다.

```text
data/backend/
.web-data/
```

주의: `.web-data`에는 기존 Python 분석 작업 파일도 있으므로 필요할 때만 삭제합니다.

## 3. Docker 없이 개발 모드로 실행

### 터미널 1 - 기존 Python 분석기

저장소 루트에서:

```powershell
.\.venv\Scripts\Activate.ps1
python -m llm_security.web.run
```

Python API:
http://127.0.0.1:8000

### 터미널 2 - .NET backend

```powershell
cd backend\LlmSecurity.Api
dotnet restore
dotnet run
```

API:
http://localhost:5080

DB:
`backend/LlmSecurity.Api/data/llm-security.db`

### 터미널 3 - React frontend

```powershell
cd frontend
npm install
npm run dev
```

웹:
http://localhost:5173

## 4. 현재 구현된 기능

- 회원가입
- 로그인 / 로그아웃
- ASP.NET Core Identity 기반 비밀번호 저장
- SQLite 사용자 데이터 저장
- 로그인 사용자별 분석 이력 분리
- 대시보드
- C/C++ 프로젝트 폴더 업로드
- 기존 Python FastAPI `/api/jobs` 호출
- 분석 상태 polling
- 완료 결과 SQLite snapshot 저장
- 취약점 핵심 정보 SQLite 정규화 저장
- 검증된 취약점 선택
- 통합 patch 생성
- patch 승인 / 거절
- 수정된 프로젝트 zip 다운로드
- React 기반 새 UI
- Docker Compose로 analyzer/backend/frontend 동시 실행

## 5. SQLite 테이블 구조

Identity가 자동 생성하는 사용자 관련 테이블:

```text
AspNetUsers
AspNetRoles
AspNetUserRoles
...
```

우리가 추가한 핵심 테이블:

```text
AnalysisJobs
AnalysisFindings
PatchBatches
```

관계:

```text
AspNetUsers
    1
    │
    N
AnalysisJobs
    │
    ├── N AnalysisFindings
    │
    └── 0..1 PatchBatches
```

각 사용자는 자신의 `AnalysisJobs`만 조회할 수 있게 백엔드에서 UserId를 검사합니다.

## 6. 왜 Python 코드를 안 지우는가

현재 기존 저장소의 FastAPI에는 이미 다음 기능이 있습니다.

```text
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/analysis
POST /api/jobs/{job_id}/patches/proposal
POST /api/jobs/{job_id}/patches/{patch_id}/approve
POST /api/jobs/{job_id}/patches/{patch_id}/reject
GET  /api/jobs/{job_id}/download
```

따라서 분석 엔진을 C#으로 재작성하지 않고,
.NET이 이 API를 호출하는 Gateway / Service Layer 역할을 합니다.

## 7. 기존 프론트는 당장 삭제하지 마세요

기존:

```text
src/llm_security/web/static/
```

은 일단 남겨둡니다.

이유:
1. Python 분석기 자체 동작 확인용으로 사용할 수 있음
2. 새 프론트 문제가 생겼을 때 비교 가능
3. 팀원의 기존 작업과 충돌을 줄임

새 플랫폼이 안정화된 뒤 별도 커밋에서 제거하는 편이 안전합니다.

## 8. 첫 커밋 추천

동작 확인 후 저장소 루트에서:

```powershell
git status
git add backend frontend compose.platform.yml PLATFORM_README.md
git commit -m "feat: add .NET SQLite web platform"
git push
```

## 9. 주의

현재 버전은 팀 프로젝트 개발/MVP 기준입니다.

실제 인터넷에 공개 배포하기 전에는 추가로:
- HTTPS 강제
- CSRF/Antiforgery 정책 검토
- 이메일 인증
- 비밀번호 재설정
- 업로드 악성 파일/압축 폭탄 정책
- 요청 rate limit
- 로그 및 감사 기록
- EF Core migration 운용
- 비밀키 secret manager 사용

등을 보강하는 것이 좋습니다.
