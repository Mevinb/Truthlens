#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STREAMLIT="$PROJECT_ROOT/.venv/bin/streamlit"

if [[ ! -x "$STREAMLIT" ]]; then
  printf '%s\n' "Streamlit was not found at $STREAMLIT" >&2
  printf '%s\n' "Install the project requirements before starting the app." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$STREAMLIT" run app.py "$@"

