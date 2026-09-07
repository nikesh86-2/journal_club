# Workflows

## Fresh setup

```bash
./scripts/setup.sh
cp .env.example .env      # fill in S2_API_KEY, ENTREZ_EMAIL, model paths
# edit config/topics.yaml and config/domains.yaml as needed
```

## Daily journal club run

```bash
./scripts/run_journal_club.sh all
```

This streams new papers, analyzes the unanalyzed ones (LLM summary, gap
analysis, critique, rubric quality scores), writes reports, and starts the
web UI at `http://localhost:5000`.

## Backfilling citations (once, after setup)

Foundational-paper detection and citation-aware ranking need `citation_count`:

```bash
python scripts/backfill_citations.py
```

## Re-scoring with the LLM rubric

Quality scores now come from an LLM rubric (`JOURNAL_CLUB_LLM_SCORING=1`).
Old cached results (pre-`analysis.v2`) are ignored automatically; re-analyze
in place with:

```bash
python scripts/run_analysis.py --all
```

## Training a new model version

```bash
./scripts/run_journal_club.sh collect-data
./scripts/run_journal_club.sh convert-dataset
./scripts/run_journal_club.sh training
```

Or gate activation on a held-out evaluation:

```bash
JOURNAL_CLUB_EVAL_BEFORE_ACTIVATE=1 ./scripts/run_journal_club.sh training
```

Then serve the merged model for analysis/recommendations:

```bash
JOURNAL_CLUB_USE_FINETUNED=1
```

Rollback if needed:

```bash
python core/model_version_tracker.py list
python core/model_version_tracker.py rollback <version_id>
```

## Rebuilding the search index

After large ingestions (or when migrating from the old JSON memory):

```bash
python scripts/build_faiss_index.py
```

## HPC (SLURM)

```bash
sbatch run_journalclub.slurm                 # full pipeline
sbatch scripts/run_analyzer_worker.slurm     # isolated analysis worker
```
