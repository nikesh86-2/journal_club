#!/bin/bash
#SBATCH --job-name=journal_club_test
#SBATCH --output=journal_club_test_%j.out
#SBATCH --error=journal_club_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load modules
module load cuda/12.4.1
module load miniforge
conda activate /mnt/scratch/fbsnpat/envs/biophysics-research-agent-new

# Set thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/scratch/fbsnpat/envs/biophysics-research-agent-new/lib:$LD_LIBRARY_PATH"

# Optimization settings
export JOURNAL_CLUB_MAX_ANALYSIS_WORKERS=2
export JOURNAL_CLUB_ENABLE_ANALYSIS_CACHE=1
export JOURNAL_CLUB_FORCE_CPU_OFFLOAD=0

echo "=== Journal Club Test Run ==="
echo "Starting at: $(date)"

# Step 1: Clean up duplicates
echo ""
echo "Step 1: Cleaning up duplicates..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
print(f'Before cleanup: {len(memory.papers)} papers')

# Count unique DOIs
unique_dois = set()
for p in memory.papers:
    doi = p.get('doi', '')
    if doi:
        unique_dois.add(doi.lower().replace(' ', ''))

print(f'Unique DOIs: {len(unique_dois)}')
print(f'Stats: {memory.get_statistics()}')
"

# Step 2: Run streaming (1 cycle only)
echo ""
echo "Step 2: Running streaming (1 cycle)..."
JC_STREAM_CYCLES=1 JC_STREAM_INTERVAL=30 bash scripts/run_journal_club.sh streaming

# Step 3: Run analysis
echo ""
echo "Step 3: Running analysis..."
bash scripts/run_journal_club.sh analysis

# Step 4: Generate reports
echo ""
echo "Step 4: Generating reports..."
bash scripts/run_journal_club.sh reports

echo ""
echo "=== Test Run Complete ==="
echo "Finished at: $(date)"