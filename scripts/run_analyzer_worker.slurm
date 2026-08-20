#!/bin/bash
# SLURM job script to run the Journal Club analyzer worker in a separate process
# Loads the model with CPU offload to avoid GPU OOMs and isolates heavy work from streaming.
# Adjust SBATCH resources (time, cpus, mem, partition) according to your cluster.

#SBATCH --job-name=jc-analyzer
#SBATCH --output=analyzer_worker-%j.out
#SBATCH --error=analyzer_worker-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail

echo "Starting analyzer worker on $(hostname)"
date

# Move to repo root (assumes job submitted from repo or set SLURM_SUBMIT_DIR)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd /scratch/fbsnpat/bot/journal_club || true
fi

module load miniforge
conda activate /scratch/fbsnpat/envs/biophysics-research-agent-new

# Reduce thread usage for libraries that spawn threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Force CPU offload for stability
export JOURNAL_CLUB_FORCE_CPU_OFFLOAD=0

# Ensure log folder exists
mkdir -p bot/journal_club/logs
mkdir -p results/analysis

echo "Running analyzer worker (CPU offload enabled)"
python3 scripts/analyzer_worker.py --memory-file cache/journal_club_memory.json --output-dir results/analysis

echo "Analyzer worker finished"
date
