#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ "${OS:-}" == "Windows_NT" ]]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
else
  VENV_PYTHON=".venv/bin/python"
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt
exec "$VENV_PYTHON" app.py
