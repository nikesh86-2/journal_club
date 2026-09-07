# Journal Club Documentation

Documentation for the Journal Club literature-analysis pipeline. This set
reflects the current codebase (SQLite memory, consolidated streaming,
LLM rubric scoring, and the closed training loop).

## Index

| Document | Covers |
|----------|--------|
| [overview.md](overview.md) | Architecture, data flow, component hierarchy |
| [configuration.md](configuration.md) | Environment variables, YAML configs, path defaults |
| [memory.md](memory.md) | SQLite memory backend (schema, queries, migration) |
| [pipeline.md](pipeline.md) | Streaming, analysis, recommendations, reports |
| [training.md](training.md) | Training data collection, QLoRA training, merging, evaluation |
| [web.md](web.md) | Flask web interface and API reference |
| [scripts.md](scripts.md) | Execution scripts and SLURM jobs |
| [workflow.md](workflow.md) | End-to-end workflows with concrete commands |

## Quick start

1. Copy `.env.example` to `.env` and fill in API keys and model paths.
2. Configure topics in `config/topics.yaml` and domains in `config/domains.yaml`.
3. Backfill citation counts for existing papers (one-off):
   ```bash
   python scripts/backfill_citations.py
   ```
4. Run the pipeline:
   ```bash
   ./scripts/run_journal_club.sh streaming     # ingest papers
   ./scripts/run_journal_club.sh analysis      # LLM analysis
   ./scripts/run_journal_club.sh reports       # markdown/JSON reports
   ./scripts/run_journal_club.sh web           # web interface
   ```

## Component map

```
config/            topics.yaml, domains.yaml, settings.yaml
core/
  config.py               path defaults + ${ENV} resolution
  literature_memory.py    SQLite storage (WAL, migration from legacy JSON)
  streaming_agent.py      literature ingestion (Europe PMC + Semantic Scholar)
  research_agent_adaptive.py  semantic search, embeddings, citation backfill
  paper_analyzer.py       LLM clients, summary/gaps/critique, rubric scoring
  recommendation_engine.py  foundational/conflicting/related papers
  report_generator.py     markdown reports
  training_data_collector.py  JSONL training examples
  training_trigger.py     threshold check + pipeline orchestration
  train_lora.py           QLoRA fine-tuning
  merge_lora.py           adapter merge + version registration
  eval_model.py           base-vs-merged model evaluation
  model_version_tracker.py  model versioning / rollback
web/app.py                Flask app + REST API
scripts/                  run/analysis/worker/FAISS/backfill scripts + SLURM
training/                 training config, dataset converter, artifacts
```
