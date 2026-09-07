# Training: Data Collection, QLoRA, Merge, Evaluation

The pipeline fine-tunes a student model (QLoRA) on its own analysis outputs
(distillation), then makes the merged model the preferred inference backend.

## Stage 1 — Data collection (`core/training_data_collector.py`)

From analyzed papers, per paper it emits instruction-tuning examples:

- **summarization** — title/abstract → summary
- **gap analysis** — title/abstract → gap JSON
- **critique** — paper + gaps → critique
- **quality scoring** — paper + gaps → score JSON
- **recommendations** — paper → foundational/conflicting/related
- **QA pairs** — LLM-generated question/answer pairs

`<think>` traces are stripped. Examples are appended to per-session JSONL
files under `training/journal_club_data/`. Collection filters by
`JOURNAL_CLUB_MIN_QUALITY_SCORE` (default 0.5).

Manual: `./scripts/run_journal_club.sh collect-data`.

## Stage 2 — Dataset conversion (`training/convert_journal_club_dataset.py`)

Renders examples in ChatML and splits at the **paper level** (a paper's
examples never straddle train/test). The split is locked in
`training/journal_club_data/locked_split.json` so re-runs stay consistent;
any pre-existing train/test overlap is repaired automatically (leaked DOIs
go to train only). Outputs `training/journal_club_hf_dataset/` (JSONL +
HuggingFace `datasets` format).

Manual: `./scripts/run_journal_club.sh convert-dataset`.

## Stage 3 — QLoRA training (`core/train_lora.py`)

4-bit NF4 quantization, LoRA (rank/alpha/dropout from YAML), gradient
checkpointing, paged AdamW 8-bit, BF16 — sized for a 6 GB GPU. Evaluation
settings from `journal_club_training_config.yaml` are wired into
`TrainingArguments` (`eval_strategy`, `eval_steps`, `load_best_model_at_end`,
`metric_for_best_model`), so the best checkpoint by eval loss is kept.

Model paths in the YAML support `${ENV_VAR:-default}` placeholders
(`JOURNAL_CLUB_BASE_MODEL`, `JOURNAL_CLUB_STUDENT_MODEL`).

## Stage 4 — Merge (`core/merge_lora.py`)

Merges the adapter into the base (student) model, saves a timestamped copy
plus the default path, validates it, registers a version, and activates it.

With `JOURNAL_CLUB_EVAL_BEFORE_ACTIVATE=1`, the merged model is evaluated
against the base model on held-out papers **before activation** — if it
regresses it is registered but not activated (rollback stays possible).

## Evaluation (`core/eval_model.py`)

Held-out papers come from memory, excluding DOIs in the locked training
split. Per model it measures:

- summary non-empty rate
- gap-analysis JSON parse rate
- average summary length
- optional LLM-judge scores (summary fidelity, gap plausibility, 1-5) via
  the llama-server

```bash
python core/eval_model.py --base /path/base --merged training/journal_club_merged_model --papers 10
```

## Model versioning (`core/model_version_tracker.py`)

Merged models are versioned under `training/merged_model_versions/` with an
`training/active_model.json` pointer. CLI: `list`, `register`, `activate`,
`rollback`, `cleanup`, `active`.

## Orchestration (`core/training_trigger.py`)

`check_and_trigger_training()`: threshold check
(`JOURNAL_CLUB_MIN_TRAIN_PAPERS`, default 200) → collect → convert → train →
merge → mark papers trained (training version + date in metadata).

Manual: `./scripts/run_journal_club.sh training` or
`./scripts/run_journal_club.sh all-with-training`.

## Inference integration

With `JOURNAL_CLUB_USE_FINETUNED=1`, `get_llm_client()` loads the merged
model first — the trained artifact is used for analysis and
recommendations, with the llama-server/local/OpenAI chain as fallback.
