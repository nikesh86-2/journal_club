# Configuration

All settings live in environment variables (see `.env.example`) plus three
YAML files. `core/config.py` is the single source of truth for path
defaults.

## Environment variables

Copy `.env.example` to `.env` and edit. The full list:

### API / personal
| Variable | Default | Purpose |
|----------|---------|---------|
| `S2_API_KEY` | — | Semantic Scholar API key (optional) |
| `ENTREZ_EMAIL` | `journal.club@example.com` | Contact email for NCBI Entrez |

### Paths
| Variable | Default | Purpose |
|----------|---------|---------|
| `JOURNAL_CLUB_LITERATURE_MEMORY_PATH` | `cache/journal_club_memory.db` | Memory database (a legacy `.json` path is migrated automatically) |
| `JOURNAL_CLUB_FAISS_INDEX_PATH` | `cache/faiss_index` | FAISS index directory |
| `JOURNAL_CLUB_WEB_PORT` | `5000` | Flask port |
| `JOURNAL_CLUB_FINETUNED_MODEL_PATH` | `training/journal_club_merged_model` | Merged LoRA model |

### LLM inference
| Variable | Default | Purpose |
|----------|---------|---------|
| `JOURNAL_CLUB_LLM_MODEL` | `gpt-4` | OpenAI fallback model name |
| `JOURNAL_CLUB_LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `JOURNAL_CLUB_LLM_MAX_TOKENS` | `2000` | Max new tokens |
| `JOURNAL_CLUB_USE_LLAMA_SERVER` | `1` | Use external llama-server |
| `JOURNAL_CLUB_LLAMA_SERVER_URL` | `http://localhost:8080` | llama-server base URL |
| `JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH` | *(empty)* | Local HuggingFace model |
| `JOURNAL_CLUB_GGUF_MODEL_PATH` | *(empty)* | GGUF model for llama-cpp-python |
| `JOURNAL_CLUB_FORCE_CPU_OFFLOAD` | `0` | Force CPU offload for local models |
| `JOURNAL_CLUB_FALLBACK_TO_BASE` | `1` | Continue down the client chain on failure |
| `JOURNAL_CLUB_RETRY_ATTEMPTS` | `3` | LLM request/parse retries |
| `JOURNAL_CLUB_MAX_ANALYSIS_WORKERS` | `4` | Parallel analysis threads |
| `JOURNAL_CLUB_ENABLE_ANALYSIS_CACHE` | `1` | Per-paper analysis cache |
| `JOURNAL_CLUB_LLM_SCORING` | `1` | LLM rubric scoring (falls back to rules) |
| `JOURNAL_CLUB_USE_FINETUNED` | `0` | Prefer the merged fine-tuned model |

### Embeddings
| Variable | Default | Purpose |
|----------|---------|---------|
| `JOURNAL_CLUB_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `JOURNAL_CLUB_EMBEDDING_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, … |

### Streaming
| Variable | Default | Purpose |
|----------|---------|---------|
| `JOURNAL_CLUB_TIME_WINDOW_MONTHS` | `0` | Recency window (`<= 0` = unlimited) |
| `JOURNAL_CLUB_ENABLE_SEMANTIC_SCHOLAR` | `1` | Query Semantic Scholar during streaming (`0` = Europe PMC only) |
| `JOURNAL_CLUB_STREAM_INTERVAL` | `30` | Seconds between cycles |
| `JOURNAL_CLUB_STREAM_BATCH_SIZE` | `20` | Candidates per query per cycle |
| `JOURNAL_CLUB_STREAM_MAX_IDLE` | `0` | Stop after N idle cycles (`0` = never) |
| `JOURNAL_CLUB_STREAM_MAX_CYCLES` | `0` | Stop after N cycles (`0` = never) |

### Training
| Variable | Default | Purpose |
|----------|---------|---------|
| `JOURNAL_CLUB_LORA_TRAIN` | `0` | Enable auto-training |
| `JOURNAL_CLUB_MIN_TRAIN_PAPERS` | `200` | Papers required before training |
| `JOURNAL_CLUB_MIN_QUALITY_SCORE` | `0.5` | Min overall quality for training data |
| `JOURNAL_CLUB_BASE_MODEL` | — | `${VAR:-default}` override for `model_name` in the training YAML |
| `JOURNAL_CLUB_STUDENT_MODEL` | — | `${VAR:-default}` override for `student_model_name` |
| `JOURNAL_CLUB_EVAL_BEFORE_ACTIVATE` | `0` | Gate model activation on held-out eval |
| `JOURNAL_CLUB_EVAL_PAPERS` | `8` | Held-out papers used for evaluation |
| `JOURNAL_CLUB_MAX_MODEL_VERSIONS` | `3` | Versioned models to retain |

## YAML configuration

### `config/topics.yaml`
Research topics. Each entry: `name`, `domain`, `description`,
`time_window_months`, `seed_queries`, and `domain_terms`
(`target_classes`, `motif_terms`, `avoid_terms`). Also editable via the web
UI (`/api/config/topics`).

### `config/domains.yaml`
Per-domain `relevance_terms`, `gap_categories`, and `quality_metrics`.
Relevance filtering requires a topic-term match **and** a domain-term match
when both are configured.

### `config/settings.yaml`
Web-UI-editable runtime settings (LLM server URL, temperature, token limit,
time window, batch size). Written by the Flask app, not read at import time.

### `training/journal_club_training_config.yaml`
QLoRA hyperparameters (model paths, LoRA rank/alpha/dropout, training
epochs/LR, eval strategy, sequence length). Model paths support
`${ENV_VAR:-default}` placeholders resolved at load time via
`core.config.resolve_env`.

## Path resolution

- The memory path comes from `core.config.DEFAULT_MEMORY_PATH`
  (`JOURNAL_CLUB_LITERATURE_MEMORY_PATH` or `cache/journal_club_memory.db`).
- The FAISS path comes from `core.config.DEFAULT_FAISS_INDEX_PATH`
  (`JOURNAL_CLUB_FAISS_INDEX_PATH` or `cache/faiss_index`).
- Passing a legacy `*.json` path to `JournalClubMemory` resolves to a
  sibling `*.db` and triggers migration.
