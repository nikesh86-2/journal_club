#!/bin/bash

# Local test script for Journal Club (no HPC/SLURM dependencies)

set -e

# Define log file
LOG_FILE=test_run_$(date +%Y%m%d_%H%M%S).log

# Redirect all output to log file
exec > >(tee -a $LOG_FILE) 2>&1

# Activate conda environment
echo "Activating conda environment: journal_club"
eval "$(conda shell.bash hook)"
conda activate journal_club


# Set thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Set optimization settings
export JOURNAL_CLUB_MAX_ANALYSIS_WORKERS=2
export JOURNAL_CLUB_ENABLE_ANALYSIS_CACHE=0
export JOURNAL_CLUB_FORCE_CPU_OFFLOAD=0
export JOURNAL_CLUB_USE_LLAMA_SERVER=1
export JOURNAL_CLUB_LLAMA_SERVER_URL=http://localhost:8080
# Prevent redundant dependency checks across pipeline stages
export DEPENDENCIES_CHECKED=1

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
papers = memory.get_all_papers()
print(f'Before cleanup: {len(papers)} papers')

# Count unique DOIs
unique_dois = {p.get('doi', '').lower().replace(' ', '') for p in papers if p.get('doi')}
print(f'Unique DOIs: {len(unique_dois)}')
print(f'Stats: {memory.get_statistics()}')

# Remove duplicates (merges analyses into the kept copy)
removed = memory.deduplicate_papers()
print(f'Removed {removed} duplicate papers')
print('Cleanup complete!')
PYTHON_SCRIPT

# Step 2: Run streaming (more cycles for more papers)
echo ""
echo "Step 2: Running streaming (5 cycles for more papers)..."
JOURNAL_CLUB_STREAM_CYCLES=5 JOURNAL_CLUB_STREAM_INTERVAL=60 bash scripts/run_journal_club.sh streaming

# Step 2.5: Post-streaming deduplication
echo ""
echo "Step 2.5: Running post-streaming deduplication..."
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
removed = memory.deduplicate_papers()
print(f'Removed {removed} duplicate papers after streaming')

# Remove misclassified papers (CCR8 GPCR and red-nucleus fMRI from RNA-Protein topic)
misclassified_dois = [
    '10.1101/2023.12.30.573730',  # CCR8 GPCR structure
    '10.1101/2023.12.31.573766',  # Red nucleus fMRI
]
removed_misclassified = memory.remove_papers_by_doi(misclassified_dois)
print(f'Removed {removed_misclassified} misclassified papers from wrong topics')
PYTHON_SCRIPT

# Step 3: Run analysis
echo ""
echo "Step 3: Running analysis..."
bash scripts/run_journal_club.sh analysis

# Step 4: Generate reports
echo ""
echo "Step 4: Generating reports..."
bash scripts/run_journal_club.sh reports

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
papers = memory.get_all_papers()
analyzed = sum(1 for p in papers if p.get('summary'))
print(f'\nAnalyzed papers: {analyzed}/{len(papers)}')
PYTHON_SCRIPT

# Step 6: Start Web / API server
echo ""
echo "Step 6: Starting web server & API on http://localhost:5000..."
# In background:
python3 web/app.py &
WEB_PID=$!
echo "Web server running with PID $WEB_PID (Access http://localhost:5000/api/stats)"

echo ""
echo "=== Test Run Complete ==="
echo "Finished at: $(date)"

