@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo === LocalLLM Airgap Installer ===
if exist bundle_stamp.json (
    echo Bundle stamp:
    type bundle_stamp.json
    echo.
) else (
    echo WARNING: bundle_stamp.json missing. Bundle integrity cannot be verified.
    echo.
)

REM Verify the installers we're about to run match the SHA-256 hashes
REM recorded at bundle build time. If they don't, the operator either
REM swapped the file or the bundle was tampered with — bail out.
if exist verify-installers.ps1 (
    echo Verifying installer integrity against bundle_stamp.json...
    powershell -ExecutionPolicy Bypass -File verify-installers.ps1
    if errorlevel 1 (
        echo ERROR: installer verification failed. Refusing to install.
        exit /b 1
    )
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
echo [3/4] Creating Python venv and installing dependencies offline...
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

echo.
echo [4/4] Smart-install: detecting hardware and configuring defaults...
REM smart_install.py inspects this machine's CPU/RAM/GPU and picks a
REM sensible DEFAULT_MODEL for the sidebar so a slow box doesn't open
REM with qwen3-coder pre-selected (or a fast box default to granite).
REM The bundle ships ALL models; this only steers the UI defaults.
python scripts\smart_install.py
if errorlevel 1 (
    echo WARNING: smart-install configuration failed; continuing with defaults.
)
cd ..

echo.
echo Install complete. Run start.bat to launch LocalLLM.
echo If hardware changes, re-run: python scripts\smart_install.py
echo (Optional) Run verify.ps1 to validate bundle integrity against SHA256SUMS.txt.
endlocal
