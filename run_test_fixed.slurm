#!/bin/bash

#SBATCH --job-name=journal_club_test
#SBATCH --output=journal_club_test_%j.out
#SBATCH --error=journal_club_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

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
export JOURNAL_CLUB_FORCE_CPU_OFFLOAD=1

echo "=== Journal Club Test Run ==="
echo "Starting at: $(date)"

# Step 1: Clean up duplicates
echo ""
echo "Step 1: Cleaning up duplicates..."
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
print(f'Before cleanup: {len(memory.memory["papers"])} papers')

# Count unique DOIs
unique_dois = set()
for p in memory.memory["papers"]:
    doi = p.get('doi', '')
    if doi:
        unique_dois.add(doi.lower().replace(' ', ''))

print(f'Unique DOIs: {len(unique_dois)}')
print(f'Stats: {memory.get_statistics()}')

# Remove duplicates
seen_dois = set()
unique_papers = []
for paper in memory.memory["papers"]:
    doi = paper.get('doi', '').lower().replace(' ', '')
    if doi and doi not in seen_dois:
        seen_dois.add(doi)
        unique_papers.append(paper)
    elif not doi:
        # Papers without DOI - keep them all for now
        unique_papers.append(paper)

print(f'After cleanup: {len(unique_papers)} papers')
memory.memory["papers"] = unique_papers
memory.update_statistics()
memory.save()
print('Cleanup complete!')
PYTHON_SCRIPT

# Step 2: Run streaming (1 cycle only)
echo ""
echo "Step 2: Running streaming (1 cycle)..."
JC_STREAM_CYCLES=1 JC_STREAM_INTERVAL=30 bash scripts/run_journal_club.sh streaming

# Step 3: Run analysis
echo ""
echo "Step 3: Running analysis..."
bash scripts/run_journal_club.sh analysis

# Step 4: Generate reports (skip if there are issues)
echo ""
echo "Step 4: Skipping reports generation (known issues)..."

# Step 5: Show final stats
echo ""
echo "Step 5: Final statistics..."
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
print(memory.summary())

# Count analyzed papers
analyzed = sum(1 for p in memory.memory["papers"] if p.get('summary'))
print(f'\nAnalyzed papers: {analyzed}/{len(memory.memory["papers"])}')
PYTHON_SCRIPT

echo ""
echo "=== Test Run Complete ==="
echo "Finished at: $(date)"