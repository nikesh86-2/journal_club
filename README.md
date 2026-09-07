# Journal Club Pipeline

A literature analysis pipeline for journal club discussions that ingests recent papers, performs gap analysis, and recommends foundational and conflicting reading.

## Features

- **Continuous Literature Ingestion**: Streams recent papers (configurable time window, default 12 months) using semantic search
- **Gap Analysis**: Identifies methodology gaps, missing controls, statistical issues, and reproducibility concerns
- **Quality Scoring**: Rates papers on methodology rigor, statistical power, and reproducibility (LLM rubric scoring with deterministic fallback)
- **Recommendations**: Suggests foundational papers, conflicting papers, and related reading
- **Configurable Domains**: Support for multiple research domains via YAML configuration
- **Web Interface**: Flask-based web interface for browsing papers and analysis
- **Markdown Reports**: Generate detailed markdown reports for topics and individual papers
- **FAISS Index**: Uses local FAISS index for semantic search
- **LoRA Fine-Tuning**: Fine-tune the model on ingested literature to improve scientific knowledge

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nikesh86-2/journal_club.git
cd journal_club
```

2. Run setup:
```bash
./scripts/setup.sh
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

### Topics (config/topics.yaml)
Define research topics with domains, seed queries, and domain-specific terms:

```yaml
topics:
  - name: "RNA-Protein Interactions"
    domain: "biophysics"
    description: "Recent advances in RNA-protein binding mechanisms"
    time_window_months: 0
    seed_queries:
      - "RNA protein binding interface"
      - "RNA-protein complex structure"
    domain_terms:
      target_classes: ["rna-binding", "ribonucleoprotein"]
      motif_terms: ["stem-loop", "hairpin"]
      avoid_terms: ["dna-binding"]
```

### Domains (config/domains.yaml)
Configure domain-specific relevance terms and analysis categories:

```yaml
domains:
  biophysics:
    relevance_terms: ["molecular dynamics", "docking", "binding affinity"]
    gap_categories: ["methodology", "controls", "statistics", "reproducibility"]
```

### Environment Variables (.env)
```bash
# personal details
# Journal club specific
S2_API_KEY=your_semantic_scholar_api_key
ENTREZ_EMAIL=your_email@example.com

# Journal club specific
JOURNAL_CLUB_FAISS_INDEX_PATH=./cache/faiss_index
JOURNAL_CLUB_TIME_WINDOW_MONTHS=0  # Set to 0 or negative for unlimited historical ingestion
JOURNAL_CLUB_MAX_MEMORY_PAPERS=1000  # Max unique papers stored in memory before quality-aware trimming
JOURNAL_CLUB_WEB_PORT=5000
JOURNAL_CLUB_LLM_MODEL=gpt-4
JOURNAL_CLUB_LLM_BACKEND=auto  # auto | finetuned | llama_server | gguf | local_hf | openai
JOURNAL_CLUB_LITERATURE_MEMORY_PATH=cache/journal_club_memory.db

# Embeddings (used for FAISS indexing and semantic search)
JOURNAL_CLUB_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
JOURNAL_CLUB_EMBEDDING_DEVICE=auto

# LLM-based quality scoring (fallback to rule-based scoring when disabled or unavailable)
JOURNAL_CLUB_LLM_SCORING=1

# LoRA fine-tuning settings
JOURNAL_CLUB_LORA_TRAIN=0
JOURNAL_CLUB_MIN_TRAIN_PAPERS=200
JOURNAL_CLUB_MIN_QUALITY_SCORE=0.5
JOURNAL_CLUB_USE_FINETUNED=0
JOURNAL_CLUB_FINETUNED_MODEL_PATH=training/journal_club_merged_model
JOURNAL_CLUB_FALLBACK_TO_BASE=1

# Evaluate the merged model vs the base model on held-out papers before activating it
JOURNAL_CLUB_EVAL_BEFORE_ACTIVATE=0
JOURNAL_CLUB_EVAL_PAPERS=8
```

## Usage

### Run Full Pipeline
```bash
./scripts/run_journal_club.sh all
```

This will:
1. Start literature streaming for all configured topics
2. Run paper analysis (gap analysis, quality scoring)
3. Generate markdown reports
4. Start the web interface

### Run Individual Components

**Streaming only:**
```bash
./scripts/run_journal_club.sh streaming
```

**Analysis only:**
```bash
./scripts/run_journal_club.sh analysis
```

**Reports only:**
```bash
./scripts/run_journal_club.sh reports
```

**Web interface only:**
```bash
./scripts/run_journal_club.sh web
```

**Training (LoRA fine-tuning):**
```bash
./scripts/run_journal_club.sh training
```

**Collect training data:**
```bash
./scripts/run_journal_club.sh collect-data
```

**Convert dataset:**
```bash
./scripts/run_journal_club.sh convert-dataset
```

**Full pipeline with training:**
```bash
./scripts/run_journal_club.sh all-with-training
```

**Backfill citation counts for existing papers:**
```bash
python scripts/backfill_citations.py
```

**Compare base vs merged model on held-out papers:**
```bash
python core/eval_model.py --base /path/to/base_model --merged training/journal_club_merged_model --papers 10
```

### Python API

```python
from core import JournalClubMemory, analyze_paper, generate_recommendations

# Initialize memory
memory = JournalClubMemory()

# Analyze a paper
paper = {"title": "...", "abstract": "..."}
analysis = analyze_paper(paper, domain="biophysics")

# Generate recommendations
recommendations = generate_recommendations(paper, all_papers)
```

## Web Interface

Access the web interface at `http://localhost:5000` (or configured port).

Features:
- **Dashboard**: Overview of all topics and statistics
- **Topic View**: Browse papers by topic with quality scores and gap analysis (paginated, 50/page)
- **Paper Detail**: Full paper information, summary, critique, and recommendations
- **Semantic Search**: Search ingested papers via the local FAISS index (`GET /api/search?q=...`)
- **Export**: Download markdown reports and JSON exports

## Output

### Markdown Reports
Generated in `output/markdown/`:
- `summary_report.md` - Overall statistics
- `{topic_name}_report.md` - Topic-specific reports
- `paper_{doi}.md` - Individual paper reports

### JSON Exports
Generated in `output/json/`:
- `{topic_name}_export.json` - Topic data with full analysis

### Literature Memory

Persistent storage in a SQLite database (`cache/journal_club_memory.db` by
default) with:
- Paper metadata and analysis results
- Gap analysis and quality scores
- Recommendation relationships
- Topic and domain tracking

Legacy JSON memory files are migrated into SQLite automatically on first
use (the JSON file is renamed with a `.migrated` suffix).

## Architecture

```
journal_club/
├── config/              # YAML configuration files
├── core/               # Core analysis modules
│   ├── literature_memory.py    # Persistent storage
│   ├── streaming_agent.py      # Literature ingestion
│   ├── paper_analyzer.py       # Gap analysis & quality scoring
│   ├── recommendation_engine.py # Foundational/conflicting detection
│   └── report_generator.py     # Markdown report generation
├── web/                # Flask web interface
│   ├── app.py
│   └── templates/
├── output/             # Generated reports
├── cache/              # FAISS index
└── scripts/            # Execution scripts
```

## Dependencies

- langchain-community (FAISS, embeddings)
- sentence-transformers (embeddings)
- requests (semantic scholar API)
- pyyaml (configuration)
- flask (web interface)
- jinja2 (templating)

## LoRA Fine-Tuning

The Journal Club pipeline supports LoRA fine-tuning to improve the model's scientific knowledge using ingested literature.

### Training Data

Training data is automatically collected from analyzed papers in the following formats:
- **Summarization**: Paper title/abstract → summary
- **Gap Analysis**: Paper → gap analysis (methodology, controls, statistics, reproducibility)
- **Critique**: Paper + gaps → structured critique
- **Quality Scoring**: Paper + gaps → quality scores
- **Recommendations**: Paper → foundational/conflicting/related papers
- **QA Pairs**: Paper → question/answer pairs

### Training Pipeline

1. **Data Collection**: When 200+ papers are analyzed, training data is collected
2. **Dataset Conversion**: JSONL data is converted to HuggingFace format
3. **LoRA Training**: Training pipeline fine-tunes the model
4. **Model Merging**: LoRA adapter is merged into base model
5. **Model Usage**: Fine-tuned model is used for analysis and recommendations

### Configuration

Enable training in `.env`:
```bash
JOURNAL_CLUB_LORA_TRAIN=1
JOURNAL_CLUB_MIN_TRAIN_PAPERS=200
JOURNAL_CLUB_MIN_QUALITY_SCORE=0.5
```

### Model Usage

When `JOURNAL_CLUB_USE_FINETUNED=1` and the merged model exists at
`JOURNAL_CLUB_FINETUNED_MODEL_PATH`, the fine-tuned model is loaded locally and
**takes priority over the external llama-server** for analysis and
recommendations. Otherwise the llama-server (or local/OpenAI fallbacks) is used.

Use fine-tuned model:
```bash
JOURNAL_CLUB_USE_FINETUNED=1
JOURNAL_CLUB_FINETUNED_MODEL_PATH=training/journal_club_merged_model
```

### Manual Training

```bash
# Collect training data
./scripts/run_journal_club.sh collect-data

# Convert to HuggingFace dataset
./scripts/run_journal_club.sh convert-dataset

# Trigger training (checks threshold)
./scripts/run_journal_club.sh training
```

### Training Configuration

Training hyperparameters are configured in `training/journal_club_training_config.yaml`:
- Base model: User configured, or mistral 7B
- LoRA rank: 16, alpha: 32, dropout: 0.1
- Training epochs: 2 (literature domain)
- Learning rate: 1.5e-5
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Running on an HPC cluster

Compute nodes are usually **offline**, so models are not downloaded on demand.
Download each model once (from a login node that has internet access) into a
shared or scratch filesystem, then point the pipeline at the local paths.

### 1. Download models to shared storage

Use a path every node can see, e.g. `/scratch/$USER/models`. For HuggingFace
models, use `--local-dir` so the path is predictable:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir /scratch/$USER/models/all-MiniLM-L6-v2

huggingface-cli download <your-llm-model-id> \
  --local-dir /scratch/$USER/models/<your-llm-model-id>
```

For GGUF files, just copy the `.gguf` to the same shared location.

### 2. Point the pipeline at the local copies

Set these in `.env` or export them in your SLURM script. Pick **one** LLM
backend (`auto | finetuned | llama_server | gguf | local_hf | openai`):

```bash
# Embeddings (FAISS index + semantic search)
JOURNAL_CLUB_EMBEDDING_MODEL=/scratch/$USER/models/all-MiniLM-L6-v2

# Option A: local HuggingFace LLM
JOURNAL_CLUB_LLM_BACKEND=local_hf
JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH=/scratch/$USER/models/<your-llm-model-id>

# Option B: local GGUF LLM (llama-cpp-python)
# JOURNAL_CLUB_LLM_BACKEND=gguf
# JOURNAL_CLUB_GGUF_MODEL_PATH=/scratch/$USER/models/<model>.gguf

# Option C: merged fine-tuned model
# JOURNAL_CLUB_LLM_BACKEND=finetuned
# JOURNAL_CLUB_FINETUNED_MODEL_PATH=/scratch/$USER/models/journal_club_merged_model

# Option D: llama-server running on a node the job can reach (no local download)
# JOURNAL_CLUB_LLM_BACKEND=llama_server
# JOURNAL_CLUB_LLAMA_SERVER_URL=http://<host>:8080
```

### 3. Point HuggingFace at the shared cache (recommended)

```bash
export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface
```

This reuses one cache across nodes and avoids re-downloading tokenizers/configs.

### 4. Run via SLURM

The repo ships example scripts (`run_journalclub.slurm`, `run_test.slurm`,
`run_test_fixed.slurm`). Edit their `#SBATCH` flags, `module load`, and
`conda activate` lines for your cluster, add the exports above, then submit:

```bash
sbatch run_journalclub.slurm
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
