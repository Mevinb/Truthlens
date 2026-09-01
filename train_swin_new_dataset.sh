#!/usr/bin/env bash
# Train SwinV2-Tiny with ImageNet pretraining on the rebuilt, verified union dataset.
# Re-running this file automatically resumes the matching tag. For example, a
# completed epoch-4 checkpoint starts at epoch 5; a mid-epoch checkpoint starts
# at its next saved batch.

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/prepared/swin_union/manifest.jsonl}"
TAG="${TAG:-swin_v2_512_newdata}"
IMG_SIZE="${IMG_SIZE:-512}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PATIENCE="${PATIENCE:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
RESUME_EVERY="${RESUME_EVERY:-2000}"
PRETRAINED="${PRETRAINED:-1}"
CLEAN_START="${CLEAN_START:-0}"
REBUILD_DATASET="${REBUILD_DATASET:-1}"
VERIFY_MODE="${VERIFY_MODE:-full}"
SKIP_DATASET_VERIFY="${SKIP_DATASET_VERIFY:-0}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-swin_v2_tiny_${IMG_SIZE}_newdata.pth}"
RESUME_CHECKPOINT="$PROJECT_DIR/models/swin_resume_${TAG}.pth"
LOG_DIR="$PROJECT_DIR/results/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${TAG}.log}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_interrupt() {
    printf '\n[launcher] Training interrupted. Re-run this script to resume from:\n  %s\n' \
        "$RESUME_CHECKPOINT"
}
trap on_interrupt INT TERM

if [[ ! -x "$PYTHON" ]]; then
    echo "[launcher] Python executable not found: $PYTHON" >&2
    echo "[launcher] Create .venv and install requirements.txt first." >&2
    exit 1
fi
if [[ "$VERIFY_MODE" != "full" && "$VERIFY_MODE" != "quick" ]]; then
    echo "[launcher] VERIFY_MODE must be 'full' or 'quick'." >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

if [[ "$CLEAN_START" == "1" && -f "$RESUME_CHECKPOINT" ]]; then
    BACKUP_NAME="${RESUME_CHECKPOINT%.pth}_backup_$(date +%Y%m%d_%H%M%S).pth"
    echo "[launcher] CLEAN_START=1: moving previous resume checkpoint to $(basename "$BACKUP_NAME")"
    mv "$RESUME_CHECKPOINT" "$BACKUP_NAME"
fi

TRAIN_CMD=(
    "$PYTHON" -u src/train_swin.py
    --manifest "$MANIFEST"
    --img-size "$IMG_SIZE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --lr "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --patience "$PATIENCE"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --progress-every "$PROGRESS_EVERY"
    --resume-every "$RESUME_EVERY"
    --tag "$TAG"
    --model-name "$MODEL_NAME"
)
if [[ "$PRETRAINED" == "0" ]]; then
    TRAIN_CMD+=(--no-pretrained)
fi
if [[ -f "$RESUME_CHECKPOINT" ]]; then
    TRAIN_CMD+=(--resume "$RESUME_CHECKPOINT")
    START_MODE="RESUME automatically from $(basename "$RESUME_CHECKPOINT")"
else
    if [[ "$PRETRAINED" == "0" ]]; then
        START_MODE="FROM SCRATCH (random Swin weights)"
    else
        START_MODE="FROM IMAGENET PRETRAINED WEIGHTS (SwinV2-T ImageNet-1K)"
    fi
fi

echo
echo "======================================================================"
echo " TruthLens Swin training — $(date --iso-8601=seconds)"
echo "======================================================================"
echo " Project             : $PROJECT_DIR"
echo " Start mode          : $START_MODE"
echo " Dataset manifest    : $MANIFEST"
echo " Rebuild dataset     : $REBUILD_DATASET"
echo " Dataset verification: $SKIP_DATASET_VERIFY (0=enabled), mode=$VERIFY_MODE"
echo " Model               : SwinV2-Tiny, ${IMG_SIZE}x${IMG_SIZE}"
echo " Initialization      : $([[ "$PRETRAINED" == "0" ]] && echo random || echo ImageNet pretrained)"
echo " Epochs (total)      : $EPOCHS"
echo " Batch / accumulation: $BATCH_SIZE / $GRAD_ACCUM (effective $((BATCH_SIZE * GRAD_ACCUM)))"
echo " LR / weight decay   : $LEARNING_RATE / $WEIGHT_DECAY"
echo " Patience / seed     : $PATIENCE / $SEED"
echo " Loader workers      : $NUM_WORKERS"
echo " Terminal detail     : every $PROGRESS_EVERY batches"
echo " Recovery snapshot   : every $RESUME_EVERY batches and every epoch"
echo " Best model          : models/$MODEL_NAME"
echo " Resume state        : $RESUME_CHECKPOINT"
echo " Log                  : $LOG_FILE"
printf ' Command              : '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n======================================================================\n\n'

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[launcher] DRY_RUN=1; no dataset or training changes were made."
    exit 0
fi

"$PYTHON" -c "import torch, torchvision, PIL, numpy, sklearn, tqdm" || {
    echo "[launcher] Required Python packages are missing. Install requirements.txt." >&2
    exit 1
}

if [[ "$REBUILD_DATASET" == "1" ]]; then
    echo "[launcher] Rebuilding the union manifest (including Grok/Aurora)..."
    "$PYTHON" -u dataset/build_swin_union_manifest.py
fi

if [[ "$SKIP_DATASET_VERIFY" != "1" ]]; then
    VERIFY_CMD=("$PYTHON" -u dataset/verify_swin_dataset.py --manifest "$MANIFEST"
        --require-corpus grok_aurora=500)
    if [[ "$VERIFY_MODE" == "quick" ]]; then
        VERIFY_CMD+=(--quick)
    fi
    echo "[launcher] Verifying the exact dataset training will read..."
    "${VERIFY_CMD[@]}"
else
    echo "[launcher] WARNING: dataset verification was explicitly skipped."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[launcher] GPU status before training:"
    nvidia-smi
else
    echo "[launcher] WARNING: nvidia-smi was not found; trainer may use CPU."
fi

echo "[launcher] Starting training. Press Ctrl+C once; re-run to continue."
if "${TRAIN_CMD[@]}"; then
    status=0
else
    status=$?
fi
echo "[launcher] Trainer exited with status $status at $(date --iso-8601=seconds)."
exit "$status"
