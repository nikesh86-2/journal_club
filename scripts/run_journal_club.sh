#!/bin/bash

# Journal Club Main Execution Script
# This script starts the literature streaming, analysis, and web interface

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to repo root
cd "$REPO_ROOT"

# Load environment variables
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# Set defaults
export JOURNAL_CLUB_FAISS_INDEX_PATH="${JOURNAL_CLUB_FAISS_INDEX_PATH:-/scratch/fbsnpat/bot/VLAB2/cache/faiss_index}"
export JOURNAL_CLUB_TIME_WINDOW_MONTHS="${JOURNAL_CLUB_TIME_WINDOW_MONTHS:-12}"
export JOURNAL_CLUB_WEB_PORT="${JOURNAL_CLUB_WEB_PORT:-5000}"
export JOURNAL_CLUB_LITERATURE_MEMORY_PATH="${JOURNAL_CLUB_LITERATURE_MEMORY_PATH:-literature_memory.json}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== Journal Club Pipeline ==="
echo "FAISS Index: $JOURNAL_CLUB_FAISS_INDEX_PATH"
echo "Time Window: $JOURNAL_CLUB_TIME_WINDOW_MONTHS months"
echo "Web Port: $JOURNAL_CLUB_WEB_PORT"
echo "Memory Path: $JOURNAL_CLUB_LITERATURE_MEMORY_PATH"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask, yaml, langchain" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Add VLAB2's PARENT to path so 'VLAB2' is importable as a namespace package
if [ -d "../VLAB2" ]; then
    VLAB2_PARENT="$(cd ../VLAB2 && cd .. && pwd)"
    export PYTHONPATH="$PYTHONPATH:$VLAB2_PARENT"
fi

# Function to start streaming (foreground, runs for a configurable number of cycles)
# Usage: start_streaming [cycles] [interval]
#   cycles  - number of stream cycles to run (0 = unlimited, default: 3)
#   interval - seconds between cycles (default: from settings or 30)
start_streaming() {
    local cycles="${1:-3}"
    local interval="${2:-30}"
    echo "Starting literature streaming (cycles=$cycles, interval=${interval}s)..."
    python3 -c "
import sys, os, time, logging
sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

from core.streaming_agent import start_all_topics, stop_all_streaming, active_streams
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
started = start_all_topics(memory)
print(f'Started {started} topic streams')

cycles = int(os.environ.get('JC_STREAM_CYCLES', '$cycles'))
interval = int(os.environ.get('JC_STREAM_INTERVAL', '$interval'))

# Keep the process alive so daemon threads can fetch papers
for i in range(cycles):
    print(f'  Stream cycle {i+1}/{cycles} — active streams: {len(active_streams())}')
    time.sleep(interval)

print('Stopping all streams...')
stop_all_streaming()
print('Streaming complete.')
"
}

# Function to run analysis
run_analysis() {
    echo "Running paper analysis..."
    python3 scripts/run_analysis.py
}

# Function to generate reports
generate_reports() {
    echo "Generating reports..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from core.literature_memory import JournalClubMemory
from core.report_generator import generate_all_reports

memory = JournalClubMemory()
reports = generate_all_reports(memory)
print(f'Generated {len(reports)} reports')
"
}

# Function to start web server
start_web() {
    echo "Starting web server on port $JOURNAL_CLUB_WEB_PORT..."
    python3 web/app.py
}

# Function to trigger training
trigger_training() {
    echo "Checking training threshold and triggering if needed..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from core.training_trigger import check_and_trigger_training

check_and_trigger_training()
"
}

# Function to collect training data
collect_training_data() {
    echo "Collecting training data from literature memory..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from core.training_data_collector import collect_training_data_from_memory

count = collect_training_data_from_memory()
print(f'Collected {count} training examples')
"
}

# Function to convert dataset
convert_dataset() {
    echo "Converting training data to HuggingFace dataset..."
    python3 training/convert_journal_club_dataset.py
}

# Main menu
case "${1:-all}" in
    streaming)
        start_streaming
        ;;
    analysis)
        run_analysis
        ;;
    reports)
        generate_reports
        ;;
    web)
        start_web
        ;;
    training)
        trigger_training
        ;;
    collect-data)
        collect_training_data
        ;;
    convert-dataset)
        convert_dataset
        ;;
    all)
        echo "Starting full pipeline..."
        start_streaming 3 30
        run_analysis
        generate_reports
        echo "Full pipeline complete. Starting web server..."
        start_web
        ;;
    all-with-training)
        echo "Starting full pipeline with training..."
        start_streaming 3 30
        run_analysis
        generate_reports
        trigger_training
        echo "Full pipeline with training complete. Starting web server..."
        start_web
        ;;
    *)
        echo "Usage: $0 [streaming|analysis|reports|web|training|collect-data|convert-dataset|all|all-with-training]"
        echo ""
        echo "Commands:"
        echo "  streaming         - Start literature streaming only"
        echo "  analysis          - Run paper analysis only"
        echo "  reports           - Generate markdown reports only"
        echo "  web               - Start web interface only"
        echo "  training          - Check threshold and trigger LoRA training"
        echo "  collect-data      - Collect training data from literature memory"
        echo "  convert-dataset   - Convert training data to HuggingFace format"
        echo "  all               - Run full pipeline (default)"
        echo "  all-with-training - Run full pipeline + training if threshold met"
        exit 1
        ;;
esac
