#!/bin/bash
#SBATCH --job-name=journal_club
#SBATCH --output=journal_club_%j.out
#SBATCH --error=journal_club_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Fixed: Use available module versions (12.4.1 and python module not needed since conda provides it)
module load cuda/12.4.1
module load miniforge
conda activate /mnt/scratch/fbsnpat/envs/biophysics-research-agent-new

# Critical: Set thread limits to avoid crashes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure CUDA libraries from conda are prioritized
export LD_LIBRARY_PATH="/scratch/fbsnpat/envs/biophysics-research-agent-new/lib:$LD_LIBRARY_PATH"

bash scripts/run_journal_club.sh all