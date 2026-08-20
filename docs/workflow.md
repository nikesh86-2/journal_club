# Journal Club Workflow

## Complete Workflow Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           JOURNAL CLUB PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • Load environment variables (.env)                                                 │
│ • Load configuration (config/topics.yaml, config/domains.yaml)                      │
│ • Initialize JournalClubMemory (literature_memory.json)                              │
│ • Setup FAISS index (shared with VLAB2 or standalone)                               │
│ • Configure LLM (VLAB2 API, fine-tuned, or local model)                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 2. LITERATURE INGESTION (Streaming Agent)                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ For each topic in config/topics.yaml:                                               │
│   • Start background thread for topic                                               │
│   • Query Semantic Scholar API with seed queries                                     │
│   • Filter by domain-specific terms (target_classes, motif_terms, avoid_terms)      │
│   • Filter by publication date (time_window_months)                                 │
│   • Deduplicate by DOI and normalized title                                          │
│   • Add to FAISS index (if new)                                                     │
│   • Store in JournalClubMemory                                                      │
│   • Repeat at configured interval (default: 30s)                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 3. PAPER ANALYSIS (Two Modes)                                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│ MODE A: Main Pipeline (run_journal_club.sh analysis)                                │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ • Load unanalyzed papers from JournalClubMemory                               │   │
│ │ • Call analyze_batch() for each topic                                         │   │
│ │ • For each paper:                                                             │   │
│ │   - Generate summary using LLM                                                │   │
│ │   - Perform gap analysis (methodology, controls, statistics, reproducibility) │   │
│ │   - Generate structured critique                                              │   │
│ │   - Calculate quality scores (methodology_rigor, statistical_power, etc.)     │   │
│ │   - Update JournalClubMemory with results                                     │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│ MODE B: Worker Mode (SLURM - analyzer_worker.py)                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ • Load local LLM model (Mistral-7B) with CPU offload                        │   │
│ │ • Load papers from cache/journal_club_memory.json                           │   │
│ │ • Process each paper sequentially:                                          │   │
│ │   - Check if already analyzed (skip if result exists)                       │   │
│ │   - Run analyze_paper() with local model                                    │   │
│ │   - Write results to results/analysis/*.json                                │   │
│ │ • Isolated process prevents OOM during model loading                         │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 4. RECOMMENDATION GENERATION                                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ For each analyzed paper:                                                             │
│   • Search FAISS for similar papers                                                 │
│   • Detect foundational papers (highly cited, older)                               │
│   • Detect conflicting papers (opposite conclusions)                                │
│   • Find related papers (similar topics, recent)                                    │
│   • Store recommendations in JournalClubMemory                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 5. REPORT GENERATION                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • Generate summary_report.md (overall statistics)                                   │
│ • Generate {topic_name}_report.md (topic-specific analysis)                         │
│ • Generate paper_{doi}.md (individual paper reports)                               │
│ • Export JSON data for web interface                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 6. TRAINING DATA COLLECTION (Optional)                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Triggered when:                                                                      │
│   • JOURNAL_CLUB_MIN_TRAIN_PAPERS threshold reached (default: 200)                   │
│   • JOURNAL_CLUB_MIN_QUALITY_SCORE threshold met (default: 0.5)                     │
│                                                                                     │
│ Process:                                                                             │
│   • Collect training examples from analyzed papers:                                  │
│     - Summarization: title/abstract → summary                                       │
│     - Gap Analysis: paper → gap analysis JSON                                       │
│     - Critique: paper + gaps → structured critique                                  │
│     - Quality Scoring: paper → quality scores                                       │
│     - Recommendations: paper → related papers                                       │
│     - QA Pairs: paper → question/answer pairs                                      │
│   • Write to training/data/journal_club_training.jsonl                               │
│   • Convert to HuggingFace dataset format                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 7. LoRA FINE-TUNING (Optional - VLAB2 Integration)                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • Use VLAB2 training infrastructure                                                 │
│ • Base model: Qwen2.5-32B-Instruct (or configured model)                            │
│ • LoRA rank: 16, alpha: 32, dropout: 0.1                                           │
│ • Training epochs: 2                                                                 │
│ • Learning rate: 1.5e-5                                                             │
│ • Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj     │
│ • Output: LoRA adapter → merged model                                               │
│ • Enable via JOURNAL_CLUB_USE_FINETUNED=1                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 8. WEB INTERFACE                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • Flask application (web/app.py)                                                     │
│ • Dashboard: Overview of topics and statistics                                       │
│ • Topic View: Browse papers by topic with quality scores                             │
│ • Paper Detail: Full analysis, critique, recommendations                             │
│ • Export: Download markdown reports and JSON exports                                 │
│ • API endpoints for programmatic access                                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

## Execution Modes

### Full Pipeline (run_journal_club.sh all)
```
Initialization → Streaming (3 cycles) → Analysis → Reports → Web Interface
```

### Full Pipeline with Training (run_journal_club.sh all-with-training)
```
Initialization → Streaming (3 cycles) → Analysis → Reports → Training Check → Web Interface
```

### Individual Components
- **streaming**: Literature ingestion only
- **analysis**: Paper analysis only
- **reports**: Report generation only
- **web**: Web interface only
- **training**: Check threshold and trigger LoRA training
- **collect-data**: Collect training data from memory
- **convert-dataset**: Convert training data to HuggingFace format

### SLURM Execution
- **Main Pipeline**: run_journalclub.slurm (GPU node, 32GB memory)
- **Analyzer Worker**: run_analyzer_worker.slurm (CPU node, isolated process)

## Key Integration Points

### VLAB2 Integration
- **FAISS Index**: Shared semantic search index
- **Semantic Search**: cached_semantic_search function
- **LLM Client**: get_llm from VLAB2.orchestration.llm
- **Training Pipeline**: VLAB2's train_lora.py and merge_lora.py
- **Model Versioning**: VLAB2's model_version_tracker.py

### External APIs
- **Semantic Scholar API**: Literature search and metadata
- **Entrez API**: PubMed integration (optional)

## Data Persistence

### JournalClubMemory (literature_memory.json)
```json
{
  "papers": [
    {
      "schema_version": "journal_club_paper.v1",
      "timestamp": "2026-08-12T00:00:00",
      "title": "...",
      "abstract": "...",
      "year": 2025,
      "doi": "...",
      "pmid": "...",
      "url": "...",
      "source": "semantic_scholar",
      "topic_name": "...",
      "domain": "...",
      "summary": "...",
      "critique": "...",
      "gap_analysis": {...},
      "quality_scores": {...},
      "recommendations": {...}
    }
  ],
  "metadata": {
    "last_updated": "...",
    "total_papers": N,
    "by_topic": {...}
  }
}
```

### Analysis Results (results/analysis/*.json)
```json
{
  "paper": {...},
  "analysis": {
    "summary": "...",
    "critique": "...",
    "gap_analysis": {...},
    "quality_scores": {...}
  }
}
```

## Error Handling and Recovery

### OOM Prevention
- **Worker Mode**: Isolated analyzer worker prevents OOM during model loading
- **CPU Offload**: Local models can be offloaded to CPU (JOURNAL_CLUB_FORCE_CPU_OFFLOAD=1)
- **Memory Limits**: SLURM jobs configured with appropriate memory limits

### Analysis Failures
- **JSON Parsing Fallback**: Rule-based gap analysis if LLM output parsing fails
- **Model Fallback**: Falls back to base model if fine-tuned model unavailable
- **Retry Logic**: Papers can be re-analyzed if initial analysis fails

### Streaming Failures
- **API Rate Limiting**: Configurable intervals between queries
- **Network Resilience**: Background threads continue on transient failures
- **Deduplication**: Prevents duplicate papers on retry
