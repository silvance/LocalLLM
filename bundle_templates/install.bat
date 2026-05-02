@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo === LocalLLM Airgap Installer ===
if exist bundle_stamp.json (
    echo Bundle stamp:
    type bundle_stamp.json
    echo.
)

echo [1/3] Installing Python (silent)...
set "PY_INSTALLER="
for %%f in (installers\python-*-amd64.exe) do set "PY_INSTALLER=%%f"
if not defined PY_INSTALLER (
    echo ERROR: No Python installer found in installers\
    echo        Place python-3.12.x-amd64.exe there before running this script.
    exit /b 1
)
echo   running %PY_INSTALLER%
"%PY_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_launcher=1
if errorlevel 1 (
    echo ERROR: Python install failed.
    exit /b 1
)

echo.
echo [2/3] Installing Ollama...
if not exist "installers\OllamaSetup.exe" (
    echo ERROR: installers\OllamaSetup.exe not found.
    exit /b 1
)
"installers\OllamaSetup.exe" /SILENT
if errorlevel 1 (
    echo ERROR: Ollama install failed.
    exit /b 1
)

REM NOTE: we deliberately do NOT copy ollama_models\ into %USERPROFILE%\.ollama\.
REM start.bat sets OLLAMA_MODELS to the bundle-local path so we never touch
REM any existing Ollama install on this machine.

echo.
echo [3/3] Creating Python venv and installing dependencies offline...
cd LocalLLM
python -m venv .venv
if errorlevel 1 (
    echo ERROR: venv creation failed. Open a fresh shell so PATH picks up Python, then re-run.
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --no-index --find-links ..\wheels --upgrade pip wheel
python -m pip install --no-index --find-links ..\wheels -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)
if not exist ".env" (
    if exist ".env.example" copy /Y ".env.example" ".env" >nul
)
cd ..

echo.
echo Install complete. Run start.bat to launch LocalLLM.
echo (Optional) Run verify.ps1 to validate bundle integrity against SHA256SUMS.txt.
endlocal
