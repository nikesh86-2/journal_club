#!/bin/bash
#SBATCH --job-name=jc-train-data
#SBATCH --output=jc_train_data_%j.out
#SBATCH --error=jc_train_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Collect LoRA training data from the analyzed literature memory and convert it
# to the HuggingFace dataset consumed by core/train_lora.py.
#
#   Stage 1  core/training_data_collector.py       -> training/journal_club_data/*.jsonl
#   Stage 2  training/convert_journal_club_dataset.py -> training/journal_club_hf_dataset/
#
# Collection needs the LLM for recommendations and QA pairs, so this runs on a
# GPU node. Submit from the repo root:  sbatch scripts/run_training_data.slurm

set -euo pipefail

module load cuda/12.4.1
module load miniforge
conda activate /mnt/scratch/fbsnpat/envs/biophysics-research-agent-new

# Thread limits (mirrors the main pipeline job).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/scratch/fbsnpat/envs/biophysics-research-agent-new/lib:${LD_LIBRARY_PATH:-}"

# Reuse the cached models (compute nodes are offline).
export HF_HOME=/users/fbsnpat/.cache/huggingface

# Run from the directory the job was submitted from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# The dataset is rendered in Qwen ChatML, and inference distils Qwen, so the
# fine-tune base MUST be the same Qwen2.5-7B-Instruct snapshot. The YAML default
# is ./mistral-7b (a base model with no chat template), which would produce an
# artifact that is not a drop-in replacement for the analysis backend.
QWEN_SNAPSHOT=/users/fbsnpat/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
if [ ! -f "$QWEN_SNAPSHOT/config.json" ]; then
    echo "ERROR: Qwen snapshot missing: $QWEN_SNAPSHOT" >&2
    exit 1
fi
export JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH="$QWEN_SNAPSHOT"
export JOURNAL_CLUB_BASE_MODEL="$QWEN_SNAPSHOT"
export JOURNAL_CLUB_LLM_BACKEND=local_hf
export JOURNAL_CLUB_USE_FINETUNED=0

# Same SQLite database the streaming/analysis pipeline writes to.
export JOURNAL_CLUB_LITERATURE_MEMORY_PATH=literature_memory.db

echo "=== Journal Club Training Data ==="
echo "Started at: $(date)"
echo "Memory Path: $JOURNAL_CLUB_LITERATURE_MEMORY_PATH"
echo "Base model:  $JOURNAL_CLUB_BASE_MODEL"
echo ""

# Stage 1: collect JSONL training examples from analyzed papers.
echo "Step 1: Collecting training data..."
bash scripts/run_journal_club.sh collect-data

DATA_DIR=training/journal_club_data
if ! compgen -G "$DATA_DIR/*.jsonl" > /dev/null; then
    echo "ERROR: no training examples were written to $DATA_DIR" >&2
    exit 1
fi
echo "Collected $(cat "$DATA_DIR"/*.jsonl | wc -l) training examples"

# Stage 2: render ChatML + paper-level train/test split, export HF dataset.
echo ""
echo "Step 2: Converting training data to HuggingFace dataset..."
bash scripts/run_journal_club.sh convert-dataset

if [ ! -f training/journal_club_hf_dataset/train.jsonl ]; then
    echo "ERROR: dataset conversion did not produce training/journal_club_hf_dataset/train.jsonl" >&2
    exit 1
fi

echo ""
echo "=== Training data ready ==="
echo "JSONL examples: $(cat "$DATA_DIR"/*.jsonl | wc -l)"
echo "HF dataset:     training/journal_club_hf_dataset"
echo "Finished at: $(date)"
