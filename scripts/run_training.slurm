#!/bin/bash
#SBATCH --job-name=jc-lora-train
#SBATCH --output=jc_lora_train_%j.out
#SBATCH --error=jc_lora_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# LoRA fine-tuning for Journal Club.
# Runs after training data has been collected and converted.
# Assumes: training/journal_club_hf_dataset/ exists with train.jsonl and test.jsonl
# If datasets library not available, will install it and create HF dataset from JSONL.

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

# Use Qwen2.5-7B-Instruct snapshot as base model (consistent with training data).
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

echo "=== Journal Club LoRA Fine-Tuning ==="
echo "Started at: $(date)"
echo "Memory Path: $JOURNAL_CLUB_LITERATURE_MEMORY_PATH"
echo "Base model:  $JOURNAL_CLUB_BASE_MODEL"
echo ""

# Ensure the HF dataset exists; if not, create it from JSONL using datasets library.
HF_DATASET_PATH="training/journal_club_hf_dataset/hf_disk"
if [ ! -d "$HF_DATASET_PATH" ]; then
    echo "HF dataset not found at $HF_DATASET_PATH. Creating from JSONL..."
    # Install datasets if not present
    pip install datasets --quiet
    # Convert JSONL to HF dataset
    python3 -c "
from datasets import DatasetDict, Dataset
import json
from pathlib import Path
import sys

data_dir = Path('training/journal_club_hf_dataset')
train_path = data_dir / 'train.jsonl'
test_path = data_dir / 'test.jsonl'

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl(train_path)
test_data = load_jsonl(test_path)

# Extract text field
train_texts = [d['text'] for d in train_data]
test_texts = [d['text'] for d in test_data]

dataset = DatasetDict({
    'train': Dataset.from_dict({'text': train_texts}),
    'test': Dataset.from_dict({'text': test_data})
})
dataset.save_to_disk(str(data_dir / 'hf_disk'))
print('Saved HF dataset to', data_dir / 'hf_disk')
"
fi

# Verify dataset
if [ ! -d "$HF_DATASET_PATH" ]; then
    echo "ERROR: HF dataset still missing at $HF_DATASET_PATH" >&2
    exit 1
fi

echo "HF dataset verified at $HF_DATASET_PATH"

# Run LoRA training
echo ""
echo "Starting LoRA training..."
python3 core/train_lora.py

echo ""
echo "=== LoRA training complete ==="
echo "Finished at: $(date)"