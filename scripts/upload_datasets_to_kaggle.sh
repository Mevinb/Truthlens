#!/usr/bin/env bash
# ==============================================================================
# TruthLens — Kaggle Dataset Upload Runner
# ==============================================================================
# Uses project virtual environment to run scripts/upload_datasets_to_kaggle.py.
#
# Examples:
#   ./scripts/upload_datasets_to_kaggle.sh --list
#   ./scripts/upload_datasets_to_kaggle.sh --group core --dry-run
#   ./scripts/upload_datasets_to_kaggle.sh --dataset evaluation
#   ./scripts/upload_datasets_to_kaggle.sh --group core
# ==============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_PYTHON="${ROOT_DIR}/.venv/bin/python3"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    VENV_PYTHON="$(command -v python3)"
  else
    echo "[Error] No python interpreter found." >&2
    exit 1
  fi
fi

exec "${VENV_PYTHON}" "${SCRIPT_DIR}/upload_datasets_to_kaggle.py" "$@"
