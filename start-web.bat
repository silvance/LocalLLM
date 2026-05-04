@echo off
setlocal

cd /d "%~dp0"

if not exist .venv (
    echo No .venv found. Run setup first:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    exit /b 1
)

call .\.venv\Scripts\activate.bat
python scripts\run_web.py %*
endlocal
