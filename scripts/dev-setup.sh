#!/usr/bin/env bash
# scripts/dev-setup.sh
# Install / update the from-source dev environment on Linux/macOS.
#
# Usage (in the LocalLLM repo root):
#     ./scripts/dev-setup.sh
#
# What it does:
#   1. Picks (or creates) a .venv/ in the repo root.
#   2. Installs requirements.txt + requirements-agent.txt.
#   3. Installs requirements-extras.txt if present.
#
# This is the "from source" counterpart to bundle_templates/install.bat.
# install.bat is for the bundled LocalLLM.exe target where deps are baked
# in; this script is for `git pull` + run-from-source where you actually
# need pip to install / update Python packages.

set -euo pipefail

# CD to repo root regardless of where the script was invoked.
cd "$(dirname "$0")/.."

NO_AGENT=0
NO_EXTRAS=0
for arg in "$@"; do
    case "$arg" in
        --no-agent)  NO_AGENT=1 ;;
        --no-extras) NO_EXTRAS=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating venv at .venv/ ..."
    python3 -m venv .venv
fi

PY=".venv/bin/python"

echo "Upgrading pip / wheel ..."
"$PY" -m pip install --upgrade pip wheel >/dev/null

echo "Installing requirements.txt ..."
"$PY" -m pip install --use-pep517 -r requirements.txt

if [ "$NO_AGENT" -eq 0 ] && [ -f requirements-agent.txt ]; then
    echo "Installing requirements-agent.txt (use --no-agent to skip) ..."
    "$PY" -m pip install --use-pep517 -r requirements-agent.txt
fi

if [ "$NO_EXTRAS" -eq 0 ] && [ -f requirements-extras.txt ]; then
    echo "Installing requirements-extras.txt (use --no-extras to skip) ..."
    "$PY" -m pip install --use-pep517 -r requirements-extras.txt
fi

echo
echo "Done. Activate the venv and run:"
echo "  source .venv/bin/activate"
echo "  python scripts/run_web.py"
