@echo off
setlocal
set "RUNTIME_DIR=%~dp0"
set "REPO_DIR=%RUNTIME_DIR%.."
set "PYTHON_BIN=python"
if exist "%REPO_DIR%\.venv\Scripts\python.exe" set "PYTHON_BIN=%REPO_DIR%\.venv\Scripts\python.exe"
"%PYTHON_BIN%" "%RUNTIME_DIR%run.py" %*
exit /b %ERRORLEVEL%

