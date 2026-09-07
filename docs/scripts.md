# Scripts

## Main runner — `scripts/run_journal_club.sh`

```bash
./scripts/run_journal_club.sh streaming     # start literature streams
./scripts/run_journal_club.sh analysis      # LLM analysis of unanalyzed papers
./scripts/run_journal_club.sh reports       # markdown reports
./scripts/run_journal_club.sh web           # Flask web interface
./scripts/run_journal_club.sh training      # threshold check + training pipeline
./scripts/run_journal_club.sh collect-data  # training examples from memory
./scripts/run_journal_club.sh convert-dataset  # JSONL -> HuggingFace dataset
./scripts/run_journal_club.sh all           # streaming -> analysis -> reports -> web
./scripts/run_journal_club.sh all-with-training
```

The runner loads `.env`, sets defaults (including
`JOURNAL_CLUB_LITERATURE_MEMORY_PATH=cache/journal_club_memory.db`), and
limits BLAS threads.

## Analysis

```bash
python scripts/run_analysis.py              # unanalyzed papers only
python scripts/run_analysis.py --all        # re-analyze everything
python scripts/run_analysis.py --limit 100  # max papers per topic
```

Uses each topic's configured domain.

## Index & data maintenance

```bash
python scripts/build_faiss_index.py         # rebuild FAISS index from memory
python scripts/backfill_citations.py        # fetch citation counts (S2 batch)
python scripts/analyzer_worker.py \
    --memory-file cache/journal_club_memory.db \
    --output-dir results/analysis           # isolated per-paper analysis worker
```

## Model evaluation

```bash
python core/eval_model.py --base /path/base --merged /path/merged --papers 10
```

## Setup

```bash
./scripts/setup.sh                          # environment setup
```

## SLURM jobs

| File | Purpose |
|------|---------|
| `run_journalclub.slurm` | Full pipeline (`all`) on a GPU node |
| `run_test.slurm` | Test run: dedup → streaming → analysis → reports |
| `run_test_fixed.slurm` | Same with CPU-offload settings |
| `scripts/run_analyzer_worker.slurm` | Standalone analyzer worker |

Local equivalents: `run_test_local.sh` (dedup → streaming → analysis →
reports → web) and the log files it produces are gitignored.
