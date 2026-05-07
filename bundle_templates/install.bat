@echo off
setlocal EnableExtensions

cd /d "%~dp0"

REM ============================================================
REM LocalLLM single-exe installer.
REM
REM The .exe is fully self-contained — Python, FastAPI, ChromaDB,
REM and Ollama are all embedded. The only thing this script does
REM is run the hardware-aware smart-install once so DEFAULT_MODEL
REM in .env reflects the actual machine. Re-runnable.
REM ============================================================

if not exist "LocalLLM.exe" (
    echo ERROR: LocalLLM.exe not found in this directory.
    echo        Make sure you copied the entire bundle, not just install.bat.
    exit /b 1
)

echo === LocalLLM Installer ===
if exist bundle_stamp.json (
    echo Bundle stamp:
    type bundle_stamp.json
    echo.
)

echo [1/1] Detecting hardware and configuring defaults...
"LocalLLM.exe" smart-install
if errorlevel 1 (
    echo WARNING: smart-install failed; LocalLLM will start with conservative defaults.
)

echo.
echo Install complete. Double-click LocalLLM.exe (or run start.bat) to launch.
echo If hardware changes later, re-run: LocalLLM.exe smart-install
endlocal
