#!/bin/bash
#SBATCH --job-name=journal_club
#SBATCH --output=journal_club_%j.out
#SBATCH --error=journal_club_%j.err
#SBATCH --time=08:00:00
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

# Point HuggingFace at cached models (avoids re-downloading)
export HF_HOME=/users/fbsnpat/.cache/huggingface

# ---------------------------------------------------------------------------
# Model: use the cached instruction-tuned Qwen2.5-7B-Instruct snapshot.
# The base Mistral-7B in ./mistral-7b has no chat template and cannot follow
# the "output JSON only" prompts, so every structured call failed and fell
# back to rule-based analysis. Qwen2.5-7B-Instruct is fully cached offline.
# NOTE: .env also sets JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH to the same path,
# because run_journal_club.sh re-sources .env and would otherwise override
# the export below.
# ---------------------------------------------------------------------------
QWEN_SNAPSHOT=/users/fbsnpat/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
if [ ! -f "$QWEN_SNAPSHOT/config.json" ]; then
    echo "ERROR: Qwen snapshot missing: $QWEN_SNAPSHOT" >&2
    exit 1
fi
export JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH="$QWEN_SNAPSHOT"
export JOURNAL_CLUB_LLM_BACKEND=local_hf

# Streaming workers must NOT run LLM analysis inline: they are daemon threads,
# and freeing the shared CUDA model while one of them is mid-generate() aborts
# the whole job ("terminate called without an active exception"). Papers are
# ingested during streaming and analysed afterwards in the dedicated analysis
# step, which runs in a single foreground process and exits cleanly.
export JOURNAL_CLUB_ANALYZE_INLINE=0
export JOURNAL_CLUB_STREAM_CYCLES=3
export JOURNAL_CLUB_STREAM_INTERVAL=30
# Sync delay between streaming and analysis to let DB writes flush
export JOURNAL_CLUB_STREAM_SYNC_DELAY=5

# Older runs cached prompt-echo / rule-fallback "analyses". Wipe the cache so
# every paper gets a fresh LLM analysis this run; the cache stays enabled
# (per .env) and is repopulated with clean results.
rm -rf cache/analysis

echo "=== Journal Club Pipeline ==="
echo "Started at: $(date)"
echo "Model: $JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH"
echo "Memory Path: literature_memory.db"
echo ""

# Step 0: Clear analyses that are prompt echoes (the "System:/Human:" bug) so
# they get re-analysed. Rows without a summary are handled by the analysis
# step below anyway; this only resets polluted rows.
echo "Step 0: Clearing prompt-echo analyses from the DB..."
python3 << 'PYTHON_SCRIPT'
import sqlite3

con = sqlite3.connect("literature_memory.db")
cur = con.cursor()
cur.execute("""
    UPDATE papers
    SET summary = NULL, critique = NULL, gap_analysis = NULL, quality_scores = NULL
    WHERE summary LIKE 'System:%' OR summary LIKE '%Human:%'
""")
con.commit()
print(f"Reset {cur.rowcount} prompt-echo analyses")
con.close()
PYTHON_SCRIPT

# Step 1: Streaming ingestion (fetch + store only, no LLM in worker threads)
echo ""
echo "Step 1: Running streaming ingestion..."
bash scripts/run_journal_club.sh streaming

# Step 1.5: Sync delay to ensure DB writes are flushed before analysis
echo ""
echo "Step 1.5: Sync delay (${JOURNAL_CLUB_STREAM_SYNC_DELAY}s) to ensure DB writes are flushed..."
sleep "$JOURNAL_CLUB_STREAM_SYNC_DELAY"

# Step 2: Post-streaming deduplication (safe, idempotent)
echo ""
echo "Step 2: Deduplicating papers..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory
memory = JournalClubMemory()
removed = memory.deduplicate_papers()
print(f'Removed {removed} duplicates')
print(memory.get_statistics())
"

# Step 3: LLM analysis pass (dedicated process; loads the model once, exits cleanly)
echo ""
echo "Step 3: Running LLM analysis..."
bash scripts/run_journal_club.sh analysis

# Step 4: Reports
echo ""
echo "Step 4: Generating reports..."
bash scripts/run_journal_club.sh reports

# Step 6: Final statistics
echo ""
echo "Step 6: Final statistics..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory
memory = JournalClubMemory()
print(memory.get_statistics())
papers = memory.get_all_papers()
analyzed = sum(1 for p in papers if p.get('summary') and not str(p.get('summary', '')).startswith('System:'))
print(f'Papers with real summaries: {analyzed}/{len(papers)}')
"

echo ""
echo "=== Pipeline complete ==="
echo "Finished at: $(date)"
