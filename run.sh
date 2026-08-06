#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STREAMLIT="$PROJECT_DIR/.venv/bin/streamlit"
APP="$PROJECT_DIR/app/app.py"

if [[ ! -x "$STREAMLIT" ]]; then
    echo "Streamlit executable not found: $STREAMLIT" >&2
    echo "Create the virtual environment and install requirements first." >&2
    exit 1
fi

# Stop only Streamlit processes serving this project's app.
pkill -f "[s]treamlit run (.*[/])?app/app\.py" 2>/dev/null || true

cd "$PROJECT_DIR"
exec "$STREAMLIT" run "$APP"
