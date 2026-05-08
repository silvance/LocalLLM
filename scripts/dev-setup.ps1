# scripts/dev-setup.ps1
# Install / update the from-source dev environment on Windows.
#
# Usage (in the LocalLLM repo root):
#     powershell -ExecutionPolicy Bypass -File scripts\dev-setup.ps1
#
# What it does:
#   1. Picks (or creates) a .venv\ in the repo root.
#   2. Installs requirements.txt + requirements-agent.txt.
#   3. Installs requirements-extras.txt if present (tree-sitter etc.).
#
# This is the "from source" counterpart to bundle_templates/install.bat.
# install.bat is for the bundled LocalLLM.exe target where deps are
# baked in by PyInstaller; this script is for `git pull` + run-from-source
# where you actually need pip to install / update Python packages.

[CmdletBinding()]
param(
    [switch]$NoAgent,
    [switch]$NoExtras
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

# 1. Resolve / create the venv
$venv = Join-Path $repo ".venv"
$pythonExe = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating venv at .venv\ ..."
    py -3 -m venv .venv
    if (-not (Test-Path $pythonExe)) {
        throw "venv creation failed; is Python 3.12 on PATH?"
    }
}

# 2. Update pip itself first — older pips trip on PEP 517 wheels we use.
Write-Host "Upgrading pip / wheel ..."
& $pythonExe -m pip install --upgrade pip wheel

# 3. Required runtime deps
Write-Host "Installing requirements.txt ..."
& $pythonExe -m pip install --use-pep517 -r requirements.txt

# 4. Agent deps (optional but recommended for /agent page)
if (-not $NoAgent -and (Test-Path "requirements-agent.txt")) {
    Write-Host "Installing requirements-agent.txt (use -NoAgent to skip) ..."
    & $pythonExe -m pip install --use-pep517 -r requirements-agent.txt
}

# 5. Extras (tree-sitter, etc.)
if (-not $NoExtras -and (Test-Path "requirements-extras.txt")) {
    Write-Host "Installing requirements-extras.txt (use -NoExtras to skip) ..."
    & $pythonExe -m pip install --use-pep517 -r requirements-extras.txt
}

Write-Host ""
Write-Host "Done. Activate the venv and run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\run_web.py"
