@echo off
setlocal EnableExtensions

cd /d "%~dp0"

REM Single-click launcher. LocalLLM.exe handles everything:
REM   * spawns its embedded Ollama on 127.0.0.1:11435 against
REM     the bundle-local ollama_models/ folder
REM   * starts the FastAPI app on a free port (8765+)
REM   * opens your default browser to it
REM
REM Closing this window stops the server.

if not exist "LocalLLM.exe" (
    echo ERROR: LocalLLM.exe not found in this directory.
    echo        Make sure you copied the entire bundle.
    exit /b 1
)

"LocalLLM.exe" %*
endlocal
