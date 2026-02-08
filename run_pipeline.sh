#!/usr/bin/env bash
#
# run_pipeline.sh — End-to-end death prediction pipeline
#
# Runs all steps from preprocessing through evaluation.
# Designed to be executed on a remote server (GPU or CPU).
#
# Usage:
#   bash run_pipeline.sh                     # default: 10k sample, 0.6B model on CPU
#   bash run_pipeline.sh --full              # use entire UKB dataset
#   bash run_pipeline.sh --full --embedding-model Qwen/Qwen3-Embedding-8B   # GPU server
#   bash run_pipeline.sh --token-mode qwen   # use Qwen for trajectory token embeddings
#   bash run_pipeline.sh --skip-preprocess   # skip data preprocessing (already done)
#   bash run_pipeline.sh --skip-delphi       # skip Delphi evaluation (no checkpoint)
#   bash run_pipeline.sh --steps 3,4,5,6,7   # run only specific steps
#
# Steps:
#   1 - Build survival dataset & disease features
#   2 - Build disease trajectory matrix
#   3 - Define shared cohort split
#   4 - Generate embedding inputs (text & trajectory) + Delphi binary data
#   5 - Compute embeddings (Qwen text & trajectory token+age)
#   6 - Train & evaluate each method
#   7 - Unified comparison
#
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
FULL_DATASET=false
SAMPLE_SIZE=10000
TOKEN_MODE="random"          # "random" or "qwen"
EMBEDDING_MODEL=""           # auto-select based on device if empty
SKIP_PREPROCESS=false
SKIP_DELPHI=false
STEPS=""                     # comma-separated step numbers, or empty for all
DEVICE=""                    # auto-detect if empty
RANDOM_STATE=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            FULL_DATASET=true
            shift ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2 ;;
        --token-mode)
            TOKEN_MODE="$2"
            shift 2 ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2 ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift ;;
        --skip-delphi)
            SKIP_DELPHI=true
            shift ;;
        --steps)
            STEPS="$2"
            shift 2 ;;
        --device)
            DEVICE="$2"
            shift 2 ;;
        --random-state)
            RANDOM_STATE="$2"
            shift 2 ;;
        -h|--help)
            head -30 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# Auto-detect device
if [[ -z "$DEVICE" ]]; then
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

# Auto-select embedding model based on device if not explicitly set
# GPU:   Qwen3-Embedding-8B  (4096-dim, best quality)
# CPU:   Qwen3-Embedding-0.6B (1024-dim, fast)
if [[ -z "$EMBEDDING_MODEL" ]]; then
    if [[ "$DEVICE" == "cuda" ]]; then
        EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
    else
        EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
    fi
fi

# Helper: check if a step should run
should_run() {
    local step="$1"
    if [[ -z "$STEPS" ]]; then
        return 0  # run all steps
    fi
    echo ",$STEPS," | grep -q ",$step,"
}

# ============================================================================
# Logging
# ============================================================================
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_separator() {
    log "============================================================"
}

# ============================================================================
# Print configuration
# ============================================================================
log_separator
log "Death Prediction Pipeline"
log_separator
log "Working directory : $SCRIPT_DIR"
log "Full dataset      : $FULL_DATASET"
log "Sample size       : $(if $FULL_DATASET; then echo 'ALL'; else echo $SAMPLE_SIZE; fi)"
log "Embedding model   : $EMBEDDING_MODEL"
log "Token mode        : $TOKEN_MODE"
log "Device            : $DEVICE"
log "Random state      : $RANDOM_STATE"
log "Skip preprocess   : $SKIP_PREPROCESS"
log "Skip Delphi       : $SKIP_DELPHI"
log "Steps             : ${STEPS:-all}"
log "Log file          : $LOG_FILE"
log_separator

# Track elapsed time
PIPELINE_START=$SECONDS

# ============================================================================
# Step 1: Build survival dataset & disease features
# ============================================================================
if should_run 1; then
    log_separator
    log "STEP 1: Build survival dataset & disease features"
    log_separator

    if $SKIP_PREPROCESS && [[ -f benchmarking/autoprognosis_survival_dataset.csv ]] && [[ -f benchmarking/disease_before60_features.csv ]]; then
        log "  Skipping (--skip-preprocess, files exist)"
    else
        SURV_ARGS="--random-state $RANDOM_STATE"
        if $FULL_DATASET; then
            SURV_ARGS="$SURV_ARGS --all"
        else
            SURV_ARGS="$SURV_ARGS --sample-size $SAMPLE_SIZE"
        fi

        log "  Running preprocess_diagnosis.py..."
        python benchmarking/preprocess_diagnosis.py 2>&1 | tee -a "$LOG_FILE"

        log "  Running preprocess_survival.py $SURV_ARGS..."
        python benchmarking/preprocess_survival.py $SURV_ARGS 2>&1 | tee -a "$LOG_FILE"
    fi

    log "  Step 1 complete."
fi

# ============================================================================
# Step 2: Build disease trajectory matrix
# ============================================================================
if should_run 2; then
    log_separator
    log "STEP 2: Build disease trajectory matrix"
    log_separator

    if $SKIP_PREPROCESS && [[ -f data/preprocessed/disease_trajectory.csv ]]; then
        log "  Skipping (--skip-preprocess, file exists)"
    else
        log "  Running generate_disease_trajectory.py..."
        python preprocessing/generate_disease_trajectory.py 2>&1 | tee -a "$LOG_FILE"
    fi

    log "  Step 2 complete."
fi

# ============================================================================
# Step 3: Define shared cohort split
# ============================================================================
if should_run 3; then
    log_separator
    log "STEP 3: Define shared cohort split"
    log_separator

    log "  Running cohort_split.py..."
    python evaluation/cohort_split.py --random-state "$RANDOM_STATE" 2>&1 | tee -a "$LOG_FILE"

    log "  Step 3 complete."
fi

# ============================================================================
# Step 4: Generate embedding inputs
# ============================================================================
if should_run 4; then
    log_separator
    log "STEP 4: Generate embedding inputs (text & trajectory) + Delphi binary"
    log_separator

    log "  [Method 3] Generating disease-history texts (with ages from trajectory)..."
    python preprocessing/natural_text_conversion.py \
        --trajectory-csv data/preprocessed/disease_trajectory.csv \
        --output-csv     data/preprocessed/text_before60.csv \
        --output-dir     data/preprocessed/text_before60 \
        2>&1 | tee -a "$LOG_FILE"

    log "  [Method 4] Generating trajectory texts..."
    python preprocessing/generate_trajectory_text.py \
        --output-csv  data/preprocessed/trajectory_before60.csv \
        --output-dir  data/preprocessed/trajectory_before60 \
        2>&1 | tee -a "$LOG_FILE"

    # Delphi binary data (aligned with cohort_split.json)
    if ! $SKIP_DELPHI; then
        log "  [Method 1] Generating Delphi binary data (train/val/test.bin)..."
        python Delphi/preprocess_delphi_binary.py \
            --trajectory-csv data/preprocessed/disease_trajectory.csv \
            --survival-csv   benchmarking/autoprognosis_survival_dataset.csv \
            --cohort-json    evaluation/cohort_split.json \
            --output-dir     Delphi/data/ukb_respiratory_data \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "  [Method 1] Skipping Delphi binary preprocessing (--skip-delphi)"
    fi

    log "  Step 4 complete."
fi

# ============================================================================
# Step 5: Compute embeddings
# ============================================================================
if should_run 5; then
    log_separator
    log "STEP 5: Compute embeddings"
    log_separator

    # Method 3: Text embeddings with Qwen3-Embedding
    log "  [Method 3] Text embedding with $EMBEDDING_MODEL..."
    QWEN_ARGS="--input-csv data/preprocessed/text_before60.csv \
               --output-dir data/preprocessed/embeddings_text \
               --tag patient \
               --model-name $EMBEDDING_MODEL"
    if [[ "$DEVICE" != "cuda" ]]; then
        QWEN_ARGS="$QWEN_ARGS --no-flash-attn"
    fi

    python embedding/qwen_embedding.py $QWEN_ARGS 2>&1 | tee -a "$LOG_FILE"

    # Method 4: Trajectory embeddings (works on CPU with random, GPU for qwen)
    log "  [Method 4] Trajectory embedding (token-mode=$TOKEN_MODE)..."
    TRAJ_ARGS="--input-csv data/preprocessed/trajectory_before60.csv \
               --output-dir data/preprocessed/embeddings_traj \
               --token-mode $TOKEN_MODE"

    if [[ "$TOKEN_MODE" == "qwen" ]] && [[ "$DEVICE" != "cuda" ]]; then
        log "  WARNING: Qwen token mode requires GPU. Falling back to random."
        TRAJ_ARGS="--input-csv data/preprocessed/trajectory_before60.csv \
                   --output-dir data/preprocessed/embeddings_traj \
                   --token-mode random"
    fi

    python embedding/trajectory_embedding.py $TRAJ_ARGS 2>&1 | tee -a "$LOG_FILE"

    log "  Step 5 complete."
fi

# ============================================================================
# Step 6: Train & evaluate each method
# ============================================================================
if should_run 6; then
    log_separator
    log "STEP 6: Train & evaluate each method"
    log_separator

    # Method 1: Delphi
    if ! $SKIP_DELPHI; then
        log "  [Method 1] Evaluating Delphi..."
        if [[ -f Delphi/Delphi-2M-respiratory/ckpt.pt ]]; then
            if [[ -f Delphi/data/ukb_respiratory_data/test.bin ]] || [[ -f Delphi/data/ukb_respiratory_data/val.bin ]]; then
                python evaluation/evaluate_delphi.py \
                    --split test \
                    --device "$DEVICE" \
                    2>&1 | tee -a "$LOG_FILE"
            else
                log "  WARNING: Delphi binary data not found."
                log "  Run step 4 first, or run: python Delphi/preprocess_delphi_binary.py"
            fi
        else
            log "  WARNING: Delphi checkpoint not found (Delphi/Delphi-2M-respiratory/ckpt.pt)"
            log "  Skipping Delphi evaluation."
        fi
    else
        log "  [Method 1] Skipping Delphi (--skip-delphi)"
    fi

    # Method 2: Benchmarking (CoxPH)
    log "  [Method 2] Training & evaluating CoxPH on binary disease features..."
    python evaluation/evaluate_benchmarking.py 2>&1 | tee -a "$LOG_FILE"

    # Method 3: Text Embedding + CoxPH
    if [[ -f data/preprocessed/embeddings_text/patient_embeddings.npz ]]; then
        log "  [Method 3] Training & evaluating CoxPH on text embeddings..."
        python evaluation/evaluate_embedding_survival.py \
            --embedding-dir data/preprocessed/embeddings_text \
            --tag patient \
            --method-name text_embedding \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "  [Method 3] Skipping (embeddings not found). Run step 5 with GPU first."
    fi

    # Method 4: Trajectory Embedding + CoxPH
    if [[ -f data/preprocessed/embeddings_traj/trajectory_embeddings.npz ]]; then
        log "  [Method 4] Training & evaluating CoxPH on trajectory embeddings..."
        python evaluation/evaluate_embedding_survival.py \
            --embedding-dir data/preprocessed/embeddings_traj \
            --tag trajectory \
            --method-name trajectory_embedding \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "  [Method 4] Skipping (embeddings not found). Run step 5 first."
    fi

    log "  Step 6 complete."
fi

# ============================================================================
# Step 7: Unified comparison
# ============================================================================
if should_run 7; then
    log_separator
    log "STEP 7: Unified comparison"
    log_separator

    python evaluation/unified_evaluation.py 2>&1 | tee -a "$LOG_FILE"

    log "  Step 7 complete."
fi

# ============================================================================
# Summary
# ============================================================================
ELAPSED=$(( SECONDS - PIPELINE_START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

log_separator
log "Pipeline finished in ${MINS}m ${SECS}s"
log_separator

# Print results if available
if [[ -f evaluation/unified_comparison.csv ]]; then
    log "Results:"
    cat evaluation/unified_comparison.csv | tee -a "$LOG_FILE"
fi

log "Full log: $LOG_FILE"
