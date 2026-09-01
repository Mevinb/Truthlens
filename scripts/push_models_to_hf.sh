#!/usr/bin/env bash
# ==============================================================================
# TruthLens — Safe Hugging Face Model Push
# ==============================================================================
# Uploads ONLY model weights to MevinBenty/truthlens-models, with strict
# filters to guarantee NO images, NO datasets, NO secrets leak at any cost.
#
# Allowlist:  *.pth, *.pkl, *.joblib, *.json (epoch indices) under models/
# Denylist:   *resume* , *backup* , *.bak*  (intermediate checkpoints),
#             *.png/*.jpg/*.jpeg/*.webp etc (images), .env, *.pem, etc
#
# Usage:
#   export HF_TOKEN="hf_xxx"          # or: hf auth login
#   ./scripts/push_models_to_hf.sh
#
# Dry-run (no upload, just validation):
#   ./scripts/push_models_to_hf.sh --dry-run
#
# Requirements:  hf CLI (huggingface_hub >= 1.0)
# ==============================================================================

set -Eeuo pipefail

REPO_ID="MevinBenty/truthlens-models"
LOCAL_MODELS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/models"
DRY_RUN=0
TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --token) TOKEN="$2"; shift 2 ;;
    --repo) REPO_ID="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [--token HF_xxx] [--repo REPO_ID]"
      exit 0
      ;;
    *) echo "[Error] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── Colour ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; RESET='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║  TruthLens — Safe HF Model Upload  (${REPO_ID})  ║${RESET}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝"

# ── 1. Fail-fast: check no images/media exist under models/ ────────────────
echo -e "\n${YELLOW}[1/5] Leak check — scanning models/ for images/media...${RESET}"
IMAGES=$(find "$LOCAL_MODELS_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.heic" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" \) 2>/dev/null | head -n 20 || true)
if [[ -n "$IMAGES" ]]; then
  echo -e "${RED}[FATAL] Images/media found under models/ — aborting to prevent leak:${RESET}"
  echo "$IMAGES"
  exit 1
fi
echo -e "${GREEN}  ✓ No images/media under models/${RESET}"

# ── 2. Fail-fast: check for secrets in staged upload ───────────────────────
echo -e "\n${YELLOW}[2/5] Leak check — scanning for secrets/tokens...${RESET}"
if grep -r -iE "hf_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|api_key.*=.*['\"][a-zA-Z0-9]{20,}" "$LOCAL_MODELS_DIR" 2>/dev/null | grep -v ".git" | head -n 5; then
  echo -e "${RED}[FATAL] Possible secret found in models/ — aborting${RESET}"
  exit 1
fi
# Also check for .env, .pem, kaggle.json that might have been dropped accidentally
SECRETS=$(find "$LOCAL_MODELS_DIR" -maxdepth 3 -type f \( -name ".env*" -o -name "kaggle.json" -o -name "*.pem" -o -name "*.key" \) 2>/dev/null | head -n 10 || true)
if [[ -n "$SECRETS" ]]; then
  echo -e "${RED}[FATAL] Secret files found under models/:${RESET}"
  echo "$SECRETS"
  exit 1
fi
echo -e "${GREEN}  ✓ No secrets detected${RESET}"

# ── 3. Inventory allowlisted files ──────────────────────────────────────────
echo -e "\n${YELLOW}[3/5] Inventory — allowlisted model files (excludes resume/backup)...${RESET}"
ALLOWLIST=$(find "$LOCAL_MODELS_DIR" -type f \( -name "*.pth" -o -name "*.pkl" -o -name "*.joblib" -o -name "*.json" \) ! -name "*resume*" ! -name "*backup*" ! -name "*.bak*" 2>/dev/null | sort || true)
COUNT=$(echo "$ALLOWLIST" | grep -c . || true)
TOTAL_SIZE=$(find "$LOCAL_MODELS_DIR" -type f \( -name "*.pth" -o -name "*.pkl" -o -name "*.joblib" -o -name "*.json" \) ! -name "*resume*" ! -name "*backup*" ! -name "*.bak*" -exec du -ch {} + 2>/dev/null | tail -n1 | awk '{print $1}' || echo "0")
echo "$ALLOWLIST" | head -n 40
if [[ $COUNT -gt 40 ]]; then echo "  ... and $((COUNT-40)) more"; fi
echo -e "  → ${GREEN}$COUNT files, total $TOTAL_SIZE${RESET} (resume/backup excluded)"
if [[ $COUNT -eq 0 ]]; then echo -e "${RED}[FATAL] No files to upload${RESET}"; exit 1; fi

# Confirm no resume/bak slipped through
if echo "$ALLOWLIST" | grep -qiE "resume|backup|\.bak"; then
  echo -e "${RED}[FATAL] Resume/backup file slipped into allowlist${RESET}"; exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo -e "\n${GREEN}[DRY-RUN] Validation passed — would upload $COUNT files ($TOTAL_SIZE) to $REPO_ID${RESET}"
  echo "  Run without --dry-run with HF_TOKEN set to actually push."
  exit 0
fi

# ── 4. Auth check ───────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/5] Auth — checking Hugging Face token...${RESET}"
if [[ -z "$TOKEN" ]]; then
  if hf auth whoami 2>&1 | grep -q "Not logged in"; then
    echo -e "${RED}[Error] Not logged in and no HF_TOKEN env var.${RESET}"
    echo "  Set HF_TOKEN or run:  hf auth login"
    echo "  Then re-run: ./scripts/push_models_to_hf.sh"
    echo "  Dry-run validation already passed — safe to push when token is ready."
    exit 1
  else
    echo -e "${GREEN}  ✓ Logged in via cached token${RESET}"
    hf auth whoami 2>&1 | head -n 5
  fi
else
  echo -e "${GREEN}  ✓ Token provided via env${RESET}"
  # Test token without leaking it
  if ! hf auth whoami --token "$TOKEN" 2>&1 | head -n 20; then
    echo -e "${YELLOW}  ! Token check returned non-zero, but will attempt upload anyway${RESET}"
  fi
fi

# ── 5. Upload with strict include/exclude ─────────────────────────────────
echo -e "\n${YELLOW}[5/5] Upload — pushing to $REPO_ID (strict filters)...${RESET}"
echo "  Local:  $LOCAL_MODELS_DIR"
echo "  Remote: $REPO_ID"
echo "  Filters: include *.pth,*.pkl,*.joblib,*.json  exclude *resume*,*backup*,*.bak*"
echo ""

# Use explicit include/exclude so even if a stray image or secret file were
# dropped into models/, hf would not upload it.
set -x
hf upload "$REPO_ID" "$LOCAL_MODELS_DIR" . \
  --repo-type model \
  --include "*.pth" \
  --include "*.pkl" \
  --include "*.joblib" \
  --include "*.json" \
  --exclude "*resume*" \
  --exclude "*backup*" \
  --exclude "*.bak*" \
  --exclude "*.png" \
  --exclude "*.jpg" \
  --exclude "*.jpeg" \
  --exclude "*.webp" \
  --exclude "*.bmp" \
  --exclude "*.tiff" \
  --exclude "*.mp4" \
  --exclude "*.env*" \
  --exclude "kaggle.json" \
  ${TOKEN:+--token "$TOKEN"} \
  --commit-message "feat: add SwinV2-Tiny union checkpoints (ep9 + improved ep1-6 + newdata, hardstyles) + epoch indices" \
  --commit-description "Allowlisted: *.pth/*.pkl/*.joblib/*.json minus resume/backup. Verified zero images/media and zero secrets. Total ~2.2G (up from 109M classical-only). Supports src/download_weights.py snapshot_download."
set +x

echo -e "\n${GREEN}✔ Upload complete.${RESET}"
echo "  Verify:  hf download $REPO_ID --dry-run"
echo "  Pull:    python src/download_weights.py"
