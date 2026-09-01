#!/usr/bin/env bash
# ==============================================================================
# TruthLens — Improved Detector Training Launcher
# ==============================================================================
# Trains SwinV2-Tiny detector with:
#   • Inverse-frequency class weighting (countering dataset imbalance)
#   • Anti-shortcut augmentations (Gaussian blur, sharpness jitter, recompress)
#   • Live terminal metrics (Real Acc, Fake Recall, Balanced Acc, ROC AUC, VRAM)
#   • Automatic threshold calibration (<=5% False Positive Rate on reals)
#   • Mid-epoch and epoch-boundary auto-resume capability
# ==============================================================================

set -Eeuo pipefail

# --- Color Definitions ---
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/prepared/swin_union/manifest.jsonl}"
TAG="${TAG:-swin_v2_512_improved}"
IMG_SIZE="${IMG_SIZE:-512}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS_TYPE="${LOSS_TYPE:-weighted_ce}"   # weighted_ce, focal, ce
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
TARGET_FPR="${TARGET_FPR:-0.05}"
PATIENCE="${PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
RESUME_EVERY="${RESUME_EVERY:-1000}"
PRETRAINED="${PRETRAINED:-1}"
INIT_WEIGHTS="${INIT_WEIGHTS:-}"
CLEAN_START="${CLEAN_START:-0}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-swin_v2_tiny_${IMG_SIZE}_improved.pth}"
RESUME_CHECKPOINT="$PROJECT_DIR/models/swin_resume_${TAG}.pth"
LOG_DIR="$PROJECT_DIR/results/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${TAG}.log}"

# Parse custom command-line overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --img-size) IMG_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --loss-type) LOSS_TYPE="$2"; shift 2 ;;
        --pretrained) PRETRAINED="1"; shift ;;
        --no-pretrained) PRETRAINED="0"; shift ;;
        --init-weights) INIT_WEIGHTS="$2"; shift 2 ;;
        --tag) TAG="$2"; RESUME_CHECKPOINT="$PROJECT_DIR/models/swin_resume_${TAG}.pth"; LOG_FILE="$LOG_DIR/train_${TAG}.log"; shift 2 ;;
        --clean-start) CLEAN_START="1"; shift ;;
        --rebuild-dataset) REBUILD_DATASET="1"; shift ;;
        --dry-run) DRY_RUN="1"; shift ;;
        --help|-h)
            echo -e "${BOLD}Usage:${RESET} ./train_improved_detector.sh [OPTIONS]"
            echo ""
            echo -e "${BOLD}Options:${RESET}"
            echo "  --img-size INT         Input image resolution (default: 512)"
            echo "  --epochs INT           Total training epochs (default: 20)"
            echo "  --batch-size INT       Per-step batch size (default: 8)"
            echo "  --grad-accum INT       Gradient accumulation steps (default: 4, effective batch = 16)"
            echo "  --lr FLOAT             Learning rate (default: 1e-4)"
            echo "  --loss-type TYPE       Loss formulation: weighted_ce (default), focal, or ce"
            echo "  --pretrained           Initialize with ImageNet-1K pretrained weights (default: enabled)"
            echo "  --no-pretrained        Initialize randomly from scratch"
            echo "  --init-weights PATH    Warm-start/fine-tune from existing model weights (.pth)"
            echo "  --tag STRING           Run identifier tag (default: swin_v2_512_improved)"
            echo "  --clean-start          Archive previous resume checkpoint and start fresh"
            echo "  --rebuild-dataset      Rebuild union dataset manifest before training"
            echo "  --dry-run              Show training setup without starting"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}[Error] Unknown argument: $1${RESET}" >&2
            echo "Run with --help for available options."
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/results/metrics"

# Log tee setup: tee output to logfile while showing real-time terminal output
exec > >(tee -a "$LOG_FILE") 2>&1

on_interrupt() {
    printf "\n${YELLOW}${BOLD}[Launcher] Training interrupted by user (Ctrl+C).${RESET}\n"
    printf "${CYAN}To resume training from this exact batch, simply run:${RESET}\n"
    printf "  ./train_improved_detector.sh --tag %s\n\n" "$TAG"
}
trap on_interrupt INT TERM

if [[ ! -x "$PYTHON" ]]; then
    echo -e "${RED}[Error] Python environment not found: $PYTHON${RESET}" >&2
    echo "Please ensure the virtual environment is set up at .venv" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# Clean start handling
if [[ "$CLEAN_START" == "1" && -f "$RESUME_CHECKPOINT" ]]; then
    BACKUP_NAME="${RESUME_CHECKPOINT%.pth}_backup_$(date +%Y%m%d_%H%M%S).pth"
    echo -e "${YELLOW}[Clean Start] Moving existing resume checkpoint to $(basename "$BACKUP_NAME")${RESET}"
    mv "$RESUME_CHECKPOINT" "$BACKUP_NAME"
fi

# Determine start mode
if [[ -f "$RESUME_CHECKPOINT" ]]; then
    START_MODE="RESUMING from $(basename "$RESUME_CHECKPOINT")"
    START_MODE_COLOR="${YELLOW}"
elif [[ -n "$INIT_WEIGHTS" ]]; then
    START_MODE="WARM START / FINE-TUNE from $(basename "$INIT_WEIGHTS")"
    START_MODE_COLOR="${CYAN}"
else
    if [[ "$PRETRAINED" == "0" ]]; then
        START_MODE="FROM SCRATCH (Random SwinV2 Weights)"
        START_MODE_COLOR="${MAGENTA}"
    else
        START_MODE="FROM IMAGENET PRETRAINED (SwinV2-T 1K Pretrained Weights)"
        START_MODE_COLOR="${GREEN}"
    fi
fi

# Banner & Hardware Diagnostic
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║${RESET}  ${BOLD}TruthLens AI — Advanced Detector Training Launcher${RESET}                              ${CYAN}║${RESET}"
echo -e "${CYAN}║${RESET}  ${BLUE}Time:${RESET} $(date '+%Y-%m-%d %H:%M:%S %Z')                                                  ${CYAN}║${RESET}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════════════╝${RESET}"

echo -e "\n${BOLD}─── System & Hardware Environment ───${RESET}"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "GPU Detected")
    echo -e "  • ${GREEN}GPU detected:${RESET} $GPU_NAME MB"
else
    echo -e "  • ${YELLOW}GPU status:${RESET} nvidia-smi not detected (running on CPU/Integrated)"
fi

PY_TORCH_INFO=$("$PYTHON" -c "import torch; print(f'PyTorch {torch.__version__} (CUDA Available: {torch.cuda.is_available()})')")
echo -e "  • ${BLUE}PyTorch:${RESET} $PY_TORCH_INFO"

echo -e "\n${BOLD}─── Training Parameters ───${RESET}"
echo -e "  • ${BOLD}Run Tag:${RESET}           ${CYAN}$TAG${RESET}"
echo -e "  • ${BOLD}Model Arch:${RESET}        SwinV2-Tiny (${IMG_SIZE}x${IMG_SIZE})"
echo -e "  • ${BOLD}Start Mode:${RESET}        ${START_MODE_COLOR}${START_MODE}${RESET}"
echo -e "  • ${BOLD}Loss Formulation:${RESET}  ${GREEN}${LOSS_TYPE}${RESET} (Balanced weights applied)"
echo -e "  • ${BOLD}Target FPR:${RESET}        ${TARGET_FPR} (Calibrated operating threshold on reals)"
echo -e "  • ${BOLD}Batch Config:${RESET}      Batch: ${BATCH_SIZE} | Accum: ${GRAD_ACCUM} (Effective batch: $((BATCH_SIZE * GRAD_ACCUM)))"
echo -e "  • ${BOLD}Epochs / LR:${RESET}       ${EPOCHS} epochs | LR: ${LEARNING_RATE} | Decay: ${WEIGHT_DECAY}"
echo -e "  • ${BOLD}Best Output:${RESET}       models/${MODEL_NAME}"
echo -e "  • ${BOLD}Resume Snapshot:${RESET}   models/$(basename "$RESUME_CHECKPOINT") (every ${RESUME_EVERY} batches)"
echo -e "  • ${BOLD}Live Logging:${RESET}      ${LOG_FILE}"

if [[ "$REBUILD_DATASET" == "1" || ! -f "$MANIFEST" ]]; then
    echo -e "\n${YELLOW}[Dataset] Preparing / Rebuilding union dataset manifest...${RESET}"
    "$PYTHON" -u dataset/build_swin_union_manifest.py
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo -e "\n${YELLOW}[Dry Run] Verification complete. Exiting without launching training.${RESET}"
    exit 0
fi

# Build Execution Command
TRAIN_CMD=(
    "$PYTHON" -u src/train_swin.py
    --manifest "$MANIFEST"
    --img-size "$IMG_SIZE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --lr "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --loss-type "$LOSS_TYPE"
    --focal-gamma "$FOCAL_GAMMA"
    --target-fpr "$TARGET_FPR"
    --patience "$PATIENCE"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --progress-every "$PROGRESS_EVERY"
    --resume-every "$RESUME_EVERY"
    --tag "$TAG"
    --model-name "$MODEL_NAME"
)

if [[ -f "$RESUME_CHECKPOINT" ]]; then
    TRAIN_CMD+=(--resume "$RESUME_CHECKPOINT")
fi

if [[ -n "$INIT_WEIGHTS" && ! -f "$RESUME_CHECKPOINT" ]]; then
    TRAIN_CMD+=(--init-weights "$INIT_WEIGHTS")
fi

if [[ "$PRETRAINED" == "0" ]]; then
    TRAIN_CMD+=(--no-pretrained)
fi

echo -e "\n${BOLD}─── Launching Training ───${RESET}"
echo -e "${GREEN}Executing:${RESET} ${TRAIN_CMD[*]}"
echo -e "${CYAN}Press Ctrl+C at any time to pause training; re-running this script will resume automatically.${RESET}\n"

if "${TRAIN_CMD[@]}"; then
    EXIT_STATUS=0
    echo -e "\n${GREEN}${BOLD}✔ Training completed successfully!${RESET}"
    echo -e "Model checkpoint ready at: ${BOLD}models/${MODEL_NAME}${RESET}"
else
    EXIT_STATUS=$?
    echo -e "\n${YELLOW}[Launcher] Trainer stopped with exit code ${EXIT_STATUS}.${RESET}"
fi

exit "$EXIT_STATUS"
