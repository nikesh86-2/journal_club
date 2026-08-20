# Journal Club Pipeline - Architecture Overview

## System Architecture

The Journal Club Pipeline is a modular literature analysis system designed for journal club discussions. It ingests recent papers, performs gap analysis, generates critiques, and recommends related reading.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Journal Club Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Streaming  │───▶│   Literature │───▶│   Analysis   │      │
│  │    Agent     │    │    Memory    │    │   Engine     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │                 │
│         ▼                   ▼                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Semantic    │    │  Persistent  │    │  Gap/Quality │      │
│  │   Search     │    │    Storage   │    │  Scoring     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                   │                 │
│                                                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Recommend   │◀───│   Critique   │    │   Training   │      │
│  │   Engine     │    │  Generator   │    │   Pipeline   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │                 │
│         ▼                   ▼                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Reports    │    │   Web UI     │    │  LoRA Model  │      │
│  │  Generator   │    │   (Flask)    │    │  (Fine-tuned)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Literature Ingestion Flow
```
Semantic Scholar API → Streaming Agent → Domain Filtering → Time Filtering → Literature Memory → FAISS Index
```

### 2. Analysis Flow (Main Pipeline)
```
Literature Memory → Paper Analyzer → LLM (Base/Fine-tuned) → Gap Analysis → Quality Scores → Critique → Memory Update
```

### 3. Analysis Flow (Worker Mode)
```
Literature Memory → Analyzer Worker (SLURM) → Local LLM (CPU offload) → Gap Analysis → Quality Scores → JSON Results → Memory Update
```

### 4. Recommendation Flow
```
Paper → Recommendation Engine → Foundational Detection → Conflict Detection → Related Search → Recommendations
```

### 5. Training Flow
```
Analyzed Papers → Training Data Collector → JSONL → Dataset Converter → HuggingFace Dataset → VLAB2 Training → LoRA Adapter → Merged Model
```

### 6. Output Flow
```
Memory → Report Generator → Markdown Reports → Web Interface → User
```

## Component Hierarchy

```
journal_club/
├── config/                    # Configuration layer
│   ├── topics.yaml           # Research topics definition
│   ├── domains.yaml          # Domain-specific settings
│   └── settings.yaml         # Pipeline settings
│
├── core/                      # Core analysis layer
│   ├── literature_memory.py   # Persistent storage
│   ├── streaming_agent.py     # Literature ingestion
│   ├── paper_analyzer.py      # Gap analysis & scoring
│   ├── recommendation_engine.py # Paper recommendations
│   ├── report_generator.py    # Report generation
│   ├── training_data_collector.py # Training data collection
│   └── training_trigger.py    # Training orchestration
│
├── web/                       # Presentation layer
│   ├── app.py                # Flask application
│   └── templates/            # HTML templates
│
├── training/                  # Training layer
│   ├── journal_club_training_config.yaml
│   └── convert_journal_club_dataset.py
│
├── scripts/                   # Execution layer
│   ├── setup.sh              # Environment setup
│   ├── run_journal_club.sh   # Main execution script
│   ├── analyzer_worker.py    # Isolated analysis worker for SLURM
│   └── run_analyzer_worker.slurm # SLURM job config for analyzer
│
└── output/                    # Output layer
    ├── markdown/             # Markdown reports
    └── json/                 # JSON exports
```

## Key Design Principles

1. **Modularity**: Each component has a single responsibility and can be used independently
2. **Configurability**: All behavior is configurable via YAML and environment variables
3. **Extensibility**: New domains, topics, and analysis types can be added without code changes
4. **Integration**: Shares infrastructure with VLAB2 (FAISS, semantic search, training)
5. **Persistence**: All data is persisted in JSON format for transparency
6. **Quality Filtering**: Only high-quality papers are used for training and recommendations
7. **Continuous Improvement**: LoRA fine-tuning enables model improvement over time

## Technology Stack

- **Language**: Python 3.8+
- **Literature Search**: Semantic Scholar API
- **Vector Storage**: FAISS (shared with VLAB2)
- **Embeddings**: Sentence Transformers
- **LLM**: Qwen2.5-32B-Instruct (shared with VLAB2)
- **Fine-tuning**: LoRA via transformers/peft (VLAB2 infrastructure)
- **Web Framework**: Flask
- **Configuration**: YAML
- **Data Format**: JSON, JSONL

## Integration Points with VLAB2

| Component | VLAB2 Component | Purpose |
|-----------|---------------|---------|
| FAISS Index | VLAB2 FAISS | Shared semantic search index |
| Semantic Search | cached_semantic_search | Paper retrieval |
| LLM | get_llm | LLM client |
| Training | train_lora.py | LoRA fine-tuning |
| Training | merge_lora.py | Model merging |
| Training | model_version_tracker.py | Model versioning |

## Performance Considerations

- **Streaming**: Background threads prevent blocking main pipeline
- **FAISS**: Vector search is O(1) for similarity queries
- **Caching**: Semantic search results are cached
- **Deduplication**: Papers are deduplicated before storage
- **Batch Processing**: Analysis can be batched for efficiency
- **Training**: Training runs asynchronously to avoid blocking
- **Worker Isolation**: Analyzer worker runs in separate process to prevent OOM during model loading
- **CPU Offload**: Local LLM models can be offloaded to CPU to reduce memory pressure

## Security Considerations

- API keys are stored in environment variables
- No sensitive data is logged
- Training data is filtered by quality score
- Model versioning provides rollback capability
- FAISS index can be shared or isolated

## Scalability

- **Horizontal**: Multiple streaming agents can run in parallel
- **Vertical**: FAISS index scales to millions of papers
- **Training**: Model can be retrained as more data is collected
- **Storage**: JSON-based storage is easily migratable
- **Web**: Flask can be deployed behind a load balancer
