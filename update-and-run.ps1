# Pull the latest code, refresh deps, and launch the LocalLLM web UI.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File C:\path\to\LocalLLM\update-and-run.ps1
#
# Flags:
#   -Streamlit       Launch the legacy Streamlit UI instead of the FastAPI web app.
#   -SkipUpdate      Skip git pull + pip install; just launch.
#   -Port <int>      Port to listen on (FastAPI only). Default 8000.

[CmdletBinding()]
param(
    [switch]$Streamlit,
    [switch]$SkipUpdate,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# Move into the repo (script's own directory) so all paths are relative.
$repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repo
Write-Host "Repo: $repo" -ForegroundColor DarkGray

if (-not $SkipUpdate) {
    Write-Step "git pull"
    git fetch origin
    git pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git pull failed (uncommitted changes? on a different branch?). Resolve and re-run." -ForegroundColor Red
        exit 1
    }
}

# Venv: create on first run, reuse afterwards.
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Step "Creating .venv"
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Step "Activating .venv"
. .\.venv\Scripts\Activate.ps1

if (-not $SkipUpdate) {
    Write-Step "pip install -r requirements.txt"
    python -m pip install --quiet --upgrade pip
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip install failed. Re-run without -SkipUpdate after fixing." -ForegroundColor Red
        exit 1
    }
}

# .env: copy from template if missing.
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Step "First-run: copying .env.example -> .env"
        Copy-Item .env.example .env
    }
}

if ($Streamlit) {
    Write-Step "Launching Streamlit (legacy)"
    streamlit run app/main.py --server.address 127.0.0.1
} else {
    Write-Step "Launching FastAPI on http://127.0.0.1:$Port"
    python scripts/run_web.py --host 127.0.0.1 --port $Port
}
