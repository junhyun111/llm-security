LLM Security - request settings integration patch

Extract this ZIP directly into the llm-security repository root and overwrite files.

Files:
- frontend/src/pages/NewAnalysisPage.tsx
- frontend/src/pages/NewAnalysisPage.css
- backend/LlmSecurity.Api/Program.cs
- backend/LlmSecurity.Api/Services/PythonAnalyzerClient.cs
- backend/LlmSecurity.Api/Controllers/RuntimeController.cs   (new)
- model_runtime/src/llm_security_runtime/api.py
- model_runtime/src/llm_security_runtime/request_service.py  (new)

After overwrite:
1) If model_runtime/.env does not exist:
   Copy-Item model_runtime/.env.example model_runtime/.env

   A user-supplied API key can now authorize that analysis request, so
   OPENROUTER_API_KEY may remain blank for testing the per-request flow.
   Keep the model names in .env because runtime startup validates them.

2) Start runtime:
   cd model_runtime
   .\run.cmd serve --host 127.0.0.1 --port 8000

3) Start backend:
   cd backend\LlmSecurity.Api
   dotnet run

4) Start frontend:
   cd frontend
   npm run dev

Security behavior:
- User API key is not saved in SQLite.
- User API key is not written to runtime job.json / analysis.json.
- Runtime keeps it only in process memory so patch generation can reuse it.
- Restarting runtime clears that per-job key.
- In production use HTTPS between browser and ASP.NET backend.
