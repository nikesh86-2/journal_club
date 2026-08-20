# Journal Club Pipeline

A literature analysis pipeline for journal club discussions that ingests recent papers, performs gap analysis, and recommends foundational and conflicting reading.

## Features

- **Continuous Literature Ingestion**: Streams recent papers (configurable time window, default 12 months) using semantic search
- **Gap Analysis**: Identifies methodology gaps, missing controls, statistical issues, and reproducibility concerns
- **Quality Scoring**: Rates papers on methodology rigor, statistical power, and reproducibility
- **Recommendations**: Suggests foundational papers, conflicting papers, and related reading
- **Configurable Domains**: Support for multiple research domains via YAML configuration
- **Web Interface**: Flask-based web interface for browsing papers and analysis
- **Markdown Reports**: Generate detailed markdown reports for topics and individual papers
- **Shared FAISS Index**: Can share FAISS index with VLAB2 or use standalone
- **LoRA Fine-Tuning**: Fine-tune the model on ingested literature to improve scientific knowledge (integrates with VLAB2's training pipeline)

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
    time_window_months: 12
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
# Shared with VLAB2
S2_API_KEY=your_semantic_scholar_api_key
ENTREZ_EMAIL=your_email@example.com

# Journal club specific
JOURNAL_CLUB_FAISS_INDEX_PATH=./cache/faiss_index
JOURNAL_CLUB_TIME_WINDOW_MONTHS=12  # Set to 0 or negative for unlimited historical ingestion
JOURNAL_CLUB_MAX_MEMORY_PAPERS=1000  # Max unique papers stored in memory before quality-aware trimming
JOURNAL_CLUB_WEB_PORT=5000
JOURNAL_CLUB_LLM_MODEL=gpt-4
JOURNAL_CLUB_LITERATURE_MEMORY_PATH=literature_memory.json

# LoRA fine-tuning settings
JOURNAL_CLUB_LORA_TRAIN=0
JOURNAL_CLUB_MIN_TRAIN_PAPERS=200
JOURNAL_CLUB_MIN_QUALITY_SCORE=0.5
JOURNAL_CLUB_USE_FINETUNED=0
JOURNAL_CLUB_FINETUNED_MODEL_PATH=training/journal_club_merged_model
JOURNAL_CLUB_FALLBACK_TO_BASE=1
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
- **Topic View**: Browse papers by topic with quality scores and gap analysis
- **Paper Detail**: Full paper information, summary, critique, and recommendations
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
Persistent storage in `literature_memory.json` with:
- Paper metadata and analysis results
- Gap analysis and quality scores
- Recommendation relationships
- Topic and domain tracking

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

## Integration with VLAB2

The journal club pipeline can share components with VLAB2:
- **FAISS Index**: Use VLAB2's FAISS index for semantic search
- **Semantic Search**: Use VLAB2's cached_semantic_search function
- **LLM Integration**: Use VLAB2's LLM utilities
- **LoRA Training**: Use VLAB2's training infrastructure for fine-tuning

Set `JOURNAL_CLUB_FAISS_INDEX_PATH` to point to your FAISS index directory (default: ./cache/faiss_index).

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
3. **LoRA Training**: VLAB2's training pipeline fine-tunes the model
4. **Model Merging**: LoRA adapter is merged into base model
5. **Model Usage**: Fine-tuned model is used for analysis and recommendations

### Configuration

Enable training in `.env`:
```bash
JOURNAL_CLUB_LORA_TRAIN=1
JOURNAL_CLUB_MIN_TRAIN_PAPERS=200
JOURNAL_CLUB_MIN_QUALITY_SCORE=0.5
```

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
- Base model: Qwen2.5-32B-Instruct (same as VLAB2)
- LoRA rank: 16, alpha: 32, dropout: 0.1
- Training epochs: 2 (literature domain)
- Learning rate: 1.5e-5
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## License

This project is part of the Virtual Lab ecosystem. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
