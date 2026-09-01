# Web Platform + model_runtime 실행 가이드

이 문서는 `feature/web-platform` 브랜치의 현재 웹 플랫폼과 `model_runtime`을 함께 실행하는 방법을 정리합니다.

## 최종 구조

```text
사용자 브라우저
  ↓
React Frontend :5173
  ↓
ASP.NET Backend :5080
  ↓
Python model_runtime :8000
  ↓
OpenRouter / LLM
  ↓
분석 결과 JSON
  ↓
ASP.NET Backend / SQLite
  ↓
React 결과 화면
```

서버는 총 3개를 실행합니다.

## 1. Python model_runtime

최초 1회:

```powershell
cd model_runtime
.\setup.cmd
Copy-Item .env.example .env
```

실행:

```powershell
.\run.cmd serve --host 127.0.0.1 --port 8000
```

확인:

```text
http://127.0.0.1:8000/api/health
```

정상이면:

```json
{"status":"ok"}
```

## 2. ASP.NET Backend

새 터미널:

```powershell
cd backend\LlmSecurity.Api
dotnet run
```

기본 분석기 주소는 `appsettings.json`의 다음 설정입니다.

```json
"Analyzer": {
  "BaseUrl": "http://127.0.0.1:8000"
}
```

## 3. React Frontend

새 터미널:

```powershell
cd frontend
npm install
npm run dev
```

접속:

```text
http://localhost:5173
```

## 웹 분석 설정

새 분석 화면에서 다음 값을 설정할 수 있습니다.

- 민감도: `0.0 ~ 1.0`
- OpenRouter API Key
- Router와 호환되는 LLM 모델

API Key는 웹 분석 요청 시 ASP.NET Backend를 거쳐 내부 Python Runtime에 전달됩니다.
현재 구현은 사용자 API Key를 SQLite DB나 `analysis.json`에 저장하지 않습니다.

민감도 `0.5`는 기존 Runtime 기본 동작과 동일합니다.

```text
Sensitivity 0.5
Candidate Gate 0.40
Validation minimum confidence 0.60
```

민감도가 높아질수록 임계값을 낮춰 더 많은 잠재 후보를 검사합니다.

## 개발용: 폴더 경로만으로 분석

회의에서 요청된 "프로젝트 폴더 위치 경로만 있으면 백엔드에서 Python 분석 실행" 기능입니다.

이 기능은 일반 사용자 업로드 API와 별도로 개발용 API로 제공합니다.

보안 제한:

- `Development` 환경에서만 활성화
- `localhost` 요청만 허용
- `RuntimeDebug:AllowedProjectRoot` 하위 폴더만 허용
- Junction / symlink 디렉터리를 따라가지 않음
- 파일 수와 총 용량 제한 적용
- 운영 환경에서는 자동 비활성화

기본 허용 루트는 저장소 최상위 폴더입니다.

설정:

```json
"RuntimeDebug": {
  "Enabled": true,
  "AllowedProjectRoot": "../..",
  "MaxFiles": 500,
  "MaxTotalMb": 100,
  "MaxSingleFileMb": 10
}
```

다른 위치의 테스트 프로젝트를 사용해야 하면
`backend/LlmSecurity.Api/appsettings.Development.json`의
`AllowedProjectRoot`만 변경합니다.

예:

```json
"AllowedProjectRoot": "C:\\Users\\cyjjh\\한이음"
```

운영 서버에서는 이 기능을 켜지 않는 것을 권장합니다.

## 개발용 API

Runtime 상태:

```text
GET /api/dev/runtime/status
```

폴더 경로 분석 생성:

```text
POST /api/dev/runtime/analyze-path
```

요청 예시:

```json
{
  "projectPath": "C:\\Users\\cyjjh\\한이음\\llm-security\\mini-vulnerable-c-project",
  "sensitivity": 0.5,
  "model": "deepseek/deepseek-v4-flash-0731",
  "apiKey": "sk-or-..."
}
```

응답에는 Python Runtime의 `jobId`, 상태 조회 주소, 분석 결과 조회 주소가 반환됩니다.

상태:

```text
GET /api/dev/runtime/jobs/{jobId}
```

결과:

```text
GET /api/dev/runtime/jobs/{jobId}/analysis
```

## PowerShell 디버그 실행 파일

저장소 루트에서:

```powershell
.\scripts\debug-analyze-path.ps1 `
  -ProjectPath "C:\Users\cyjjh\한이음\llm-security\mini-vulnerable-c-project" `
  -ApiKey "sk-or-..." `
  -Wait
```

`.env`에 API Key가 이미 있고 `RUN_PAID_EXPERIMENTS=1`을 사용하는 경우에는 `-ApiKey`를 생략할 수 있습니다.

민감도 변경:

```powershell
.\scripts\debug-analyze-path.ps1 `
  -ProjectPath "C:\Users\cyjjh\한이음\llm-security\mini-vulnerable-c-project" `
  -Sensitivity 0.8 `
  -ApiKey "sk-or-..." `
  -Wait
```

이 명령은:

```text
프로젝트 경로
  ↓
ASP.NET Backend
  ↓
Backend가 허용된 프로젝트 파일 수집
  ↓
HTTP multipart
  ↓
Python model_runtime
  ↓
OpenRouter
  ↓
상태 / 결과를 다시 Backend를 통해 조회
```

순서로 동작합니다.

ASP.NET Backend가 Python 프로세스를 직접 `subprocess` 방식으로 실행하지 않는 이유는,
회의에서 정한 3서버 구조를 유지하고 Python Runtime을 독립 서비스로 고정하기 위해서입니다.

## 커밋 전 확인

세 서버를 모두 켠 뒤:

```text
Python  : http://127.0.0.1:8000
Backend : http://localhost:5080
Frontend: http://localhost:5173
```

웹 업로드 분석이 정상인지 한 번 확인하고, 개발용 경로 분석도 한 번 실행합니다.

권장 커밋 메시지:

```text
feat: complete runtime integration and local path debug API
```
