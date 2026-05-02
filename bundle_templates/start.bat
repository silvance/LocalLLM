@echo off
setlocal

cd /d "%~dp0\LocalLLM"

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: venv missing. Run install.bat first.
    exit /b 1
)

REM Ollama auto-starts on Windows after install. If the chat fails to reach it,
REM open a separate shell and run: ollama serve

streamlit run app\main.py
endlocal
