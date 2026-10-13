#!/usr/bin/env bash
set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$FRONTEND_DIR/.." && pwd)"

cd "$FRONTEND_DIR"
"$PROJECT_DIR/.venv/bin/python" -m uvicorn api.server:app --reload --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

npm run dev
