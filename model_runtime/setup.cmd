@echo off
setlocal

rem model_runtime directory (%~dp0 always ends with a backslash)
set "RUNTIME_DIR=%~dp0"
if "%RUNTIME_DIR:~-1%"=="\" set "RUNTIME_DIR=%RUNTIME_DIR:~0,-1%"

set "REPO_DIR=%RUNTIME_DIR%\.."
set "PYTHON_BIN=python"

rem Prefer the repository virtual environment when it exists
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%REPO_DIR%\.venv\Scripts\python.exe"
)

echo [model_runtime] Runtime directory: "%RUNTIME_DIR%"
echo [model_runtime] Python: "%PYTHON_BIN%"
echo [model_runtime] Installing runtime package...

"%PYTHON_BIN%" -m pip install -e "%RUNTIME_DIR%"

if errorlevel 1 (
    echo.
    echo [model_runtime] Setup failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [model_runtime] Setup completed successfully.
exit /b 0
