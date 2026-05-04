@echo off
setlocal EnableExtensions

cd /d "%~dp0"

REM Bundle-local Ollama instance: doesn't touch the user's existing Ollama
REM models or service. Runs on a non-default port to avoid colliding with
REM any other Ollama install on this machine.
set "OLLAMA_MODELS=%CD%\ollama_models"
set "OLLAMA_HOST=127.0.0.1:11435"

where ollama >nul 2>&1
if errorlevel 1 (
    echo ERROR: ollama.exe not on PATH. Run install.bat first or open a fresh shell.
    exit /b 1
)

echo Starting bundle-local Ollama (port 11435, models from bundle\ollama_models)...
start "ollama-localllm" /B cmd /c "ollama serve > ollama-localllm.log 2>&1"

REM Preflight: poll /api/tags for up to 20 seconds
set /a tries=0
:waitloop
curl -s http://127.0.0.1:11435/api/tags >nul 2>&1
if not errorlevel 1 goto :ready
set /a tries+=1
if %tries% gtr 20 (
    echo ERROR: Ollama did not come up within 20 seconds.
    echo Check ollama-localllm.log for details.
    exit /b 1
)
timeout /t 1 /nobreak >nul
goto :waitloop
:ready

REM Make the same OLLAMA_HOST available to the Python app (overrides .env).
set "OLLAMA_HOST=http://127.0.0.1:11435"

cd LocalLLM
if not exist ".env" (
    if exist ".env.example" copy /Y ".env.example" ".env" >nul
)
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: venv missing. Run install.bat first.
    exit /b 1
)

REM Launch the FastAPI web UI on 127.0.0.1:8000 (single-user local tool;
REM the FastAPI app keeps generations running across tab switches /
REM browser refreshes, which the legacy Streamlit UI couldn't do).
REM
REM Pass --streamlit if you want the legacy UI:
REM   start.bat --streamlit
if /I "%~1"=="--streamlit" (
    echo Launching legacy Streamlit UI...
    streamlit run app\main.py --server.address 127.0.0.1 --server.headless true
) else (
    echo Launching FastAPI UI on http://127.0.0.1:8000 ...
    python scripts\run_web.py --host 127.0.0.1 --port 8000
)
endlocal
