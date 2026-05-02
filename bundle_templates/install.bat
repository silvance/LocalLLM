@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo === LocalLLM Airgap Installer ===
echo.

echo [1/4] Installing Python (silent)...
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
echo [2/4] Installing Ollama...
if not exist "installers\OllamaSetup.exe" (
    echo ERROR: installers\OllamaSetup.exe not found.
    exit /b 1
)
"installers\OllamaSetup.exe" /SILENT
if errorlevel 1 (
    echo ERROR: Ollama install failed.
    exit /b 1
)

echo.
echo [3/4] Restoring Ollama models to %%USERPROFILE%%\.ollama\models ...
if not exist "%USERPROFILE%\.ollama\models" mkdir "%USERPROFILE%\.ollama\models"
xcopy /E /I /Y /Q "ollama_models\*" "%USERPROFILE%\.ollama\models\" >nul
if errorlevel 1 (
    echo ERROR: model copy failed.
    exit /b 1
)

echo.
echo [4/4] Creating Python venv and installing dependencies offline...
cd LocalLLM
python -m venv .venv
if errorlevel 1 (
    echo ERROR: venv creation failed. Make sure Python is on PATH (open a new shell).
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --no-index --find-links ..\wheels --upgrade pip wheel
python -m pip install --no-index --find-links ..\wheels -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)
cd ..

echo.
echo Install complete. Run start.bat to launch LocalLLM.
endlocal
