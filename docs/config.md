# Configuration Documentation

## Overview

The Journal Club pipeline uses YAML configuration files for topics, domains, and settings. These files allow customization without code changes.

## Configuration Files

### `config/topics.yaml`

Defines research topics with their associated domains, queries, and domain-specific terms.

#### Structure

```yaml
topics:
  - name: "Topic Name"
    domain: "domain_name"
    description: "Topic description"
    time_window_months: 60
    seed_queries:
      - "Query 1"
      - "Query 2"
    domain_terms:
      target_classes: ["term1", "term2"]
      motif_terms: ["term1", "term2"]
      avoid_terms: ["term1", "term2"]
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Topic name for display and categorization |
| `domain` | string | Yes | Domain name (must exist in domains.yaml) |
| `description` | string | No | Human-readable description |
| `time_window_months` | integer | No | Time window for recent papers (default: 12) |
| `seed_queries` | list | Yes | Search queries for literature ingestion |
| `domain_terms.target_classes` | list | No | Target class terms for relevance filtering |
| `domain_terms.motif_terms` | list | No | Motif terms for relevance filtering |
| `domain_terms.avoid_terms` | list | No | Terms to exclude (negative filtering) |

#### Example

```yaml
topics:
  - name: "RNA-Protein Interactions"
    domain: "biophysics"
    description: "Recent advances in RNA-protein binding mechanisms"
    time_window_months: 60
    seed_queries:
      - "RNA protein binding interface"
      - "RNA-protein complex structure"
      - "ribonucleoprotein assembly"
    domain_terms:
      target_classes: ["rna-binding", "ribonucleoprotein", "RNP"]
      motif_terms: ["stem-loop", "hairpin", "pseudoknot"]
      avoid_terms: ["dna-binding", "protein-protein", "membrane"]
```

### `config/domains.yaml`

Defines domain-specific settings including relevance terms, gap categories, and quality metrics.

#### Structure

```yaml
domains:
  domain_name:
    relevance_terms: ["term1", "term2"]
    gap_categories:
      - "category1"
      - "category2"
    quality_metrics:
      - "metric1"
      - "metric2"
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `relevance_terms` | list | Yes | Terms that indicate domain relevance |
| `gap_categories` | list | Yes | Gap analysis categories for this domain |
| `quality_metrics` | list | No | Quality metrics for this domain |

#### Example

```yaml
domains:
  biophysics:
    relevance_terms:
      - "molecular dynamics"
      - "docking"
      - "binding affinity"
      - "structural biology"
      - "biophysical methods"
    gap_categories:
      - "methodology"
      - "controls"
      - "statistics"
      - "reproducibility"
    quality_metrics:
      - "methodology_rigor"
      - "statistical_power"
      - "reproducibility_score"
      - "control_quality"

  virology:
    relevance_terms:
      - "viral"
      - "virus"
      - "infection"
      - "pathogen"
      - "capsid"
    gap_categories:
      - "methodology"
      - "controls"
      - "statistics"
      - "reproducibility"
    quality_metrics:
      - "methodology_rigor"
      - "statistical_power"
      - "reproducibility_score"
      - "control_quality"

  general:
    relevance_terms:
      - "research"
      - "study"
      - "analysis"
      - "experiment"
    gap_categories:
      - "methodology"
      - "controls"
      - "statistics"
      - "reproducibility"
    quality_metrics:
      - "methodology_rigor"
      - "statistical_power"
      - "reproducibility_score"
      - "control_quality"
```

### `config/settings.yaml`

Pipeline-wide settings for streaming, analysis, output, and training.

#### Structure

```yaml
# Time window settings
default_time_window_months: 60

# Streaming settings
stream_interval_seconds: 30
stream_batch_size: 20
stream_max_idle_cycles: 0
stream_max_cycles: 0

# Deduplication settings
dedup_abstract_prefix_len: 500

# Analysis settings
enable_gap_analysis: true
enable_recommendations: true
enable_quality_scoring: true

# Output settings
generate_markdown_reports: true
generate_json_exports: true
web_interface_enabled: true

# LLM settings
llm_temperature: 0.3
llm_max_tokens: 2000

# LoRA fine-tuning settings
training:
  enabled: false
  min_papers_threshold: 200
  auto_trigger: true
  min_quality_score: 0.5
  model_path: /path/to/model
  output_dir: training/journal_club_output

# Fine-tuned model usage
model:
  use_finetuned: false
  finetuned_model_path: training/journal_club_merged_model
  fallback_to_base: true
```

#### Fields

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| Time Window | `default_time_window_months` | int | 12 | Default time window for recent papers |
| Streaming | `stream_interval_seconds` | int | 30 | Seconds between fetch cycles |
| Streaming | `stream_batch_size` | int | 20 | Papers per fetch |
| Streaming | `stream_max_idle_cycles` | int | 0 | Max idle cycles before stop (0 = infinite) |
| Streaming | `stream_max_cycles` | int | 0 | Max total cycles (0 = infinite) |
| Deduplication | `dedup_abstract_prefix_len` | int | 500 | Characters for abstract-based dedup |
| Analysis | `enable_gap_analysis` | bool | true | Enable gap analysis |
| Analysis | `enable_recommendations` | bool | true | Enable recommendations |
| Analysis | `enable_quality_scoring` | bool | true | Enable quality scoring |
| Output | `generate_markdown_reports` | bool | true | Generate markdown reports |
| Output | `generate_json_exports` | bool | true | Generate JSON exports |
| Output | `web_interface_enabled` | bool | true | Enable web interface |
| LLM | `llm_temperature` | float | 0.3 | LLM temperature |
| LLM | `llm_max_tokens` | int | 2000 | Max tokens per response |
| Training | `enabled` | bool | false | Enable LoRA training |
| Training | `min_papers_threshold` | int | 200 | Minimum papers for training |
| Training | `auto_trigger` | bool | true | Auto-trigger training |
| Training | `min_quality_score` | float | 0.5 | Minimum quality score for training |
| Training | `model_path` | str | - | Path to base model |
| Training | `output_dir` | str | - | Training output directory |
| Model | `use_finetuned` | bool | false | Use fine-tuned model |
| Model | `finetuned_model_path` | str | - | Path to fine-tuned model |
| Model | `fallback_to_base` | bool | true | Fall back to base model |

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `S2_API_KEY` | Semantic Scholar API key | `your_api_key_here` |
| `ENTREZ_EMAIL` | Email for Entrez API | `your_email@example.com` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_FAISS_INDEX_PATH` | `cache/faiss_index` | Path to FAISS index |
| `JOURNAL_CLUB_TIME_WINDOW_MONTHS` | `12` | Time window in months (set to 0 or negative for unlimited/historical ingestion) |
| `JOURNAL_CLUB_MAX_MEMORY_PAPERS` | `1000` | Maximum papers kept in persistent memory before quality-aware pruning |
| `JOURNAL_CLUB_WEB_PORT` | `5000` | Web server port |
| `JOURNAL_CLUB_LLM_MODEL` | `gpt-4` | LLM model name |
| `JOURNAL_CLUB_LITERATURE_MEMORY_PATH` | `literature_memory.json` | Path to memory file |
| `JOURNAL_CLUB_LORA_TRAIN` | `0` | Enable LoRA training |
| `JOURNAL_CLUB_MIN_TRAIN_PAPERS` | `200` | Minimum papers for training |
| `JOURNAL_CLUB_MIN_QUALITY_SCORE` | `0.5` | Minimum quality score |
| `JOURNAL_CLUB_USE_FINETUNED` | `0` | Use fine-tuned model |
| `JOURNAL_CLUB_FINETUNED_MODEL_PATH` | `training/journal_club_merged_model` | Path to fine-tuned model |
| `JOURNAL_CLUB_FALLBACK_TO_BASE` | `1` | Fall back to base model |

## Configuration Loading

### Loading Topics

```python
import yaml

with open("config/topics.yaml") as f:
    config = yaml.safe_load(f)

topics = config["topics"]
for topic in topics:
    print(topic["name"])
```

### Loading Domains

```python
import yaml

with open("config/domains.yaml") as f:
    config = yaml.safe_load(f)

domains = config["domains"]
for domain_name, domain_config in domains.items():
    print(domain_name)
```

### Loading Settings

```python
import yaml

with open("config/settings.yaml") as f:
    settings = yaml.safe_load(f)

time_window = settings["default_time_window_months"]
stream_interval = settings["stream_interval_seconds"]
```

## Configuration Best Practices

### 1. Domain Consistency
Ensure domain names in `topics.yaml` match domain names in `domains.yaml`.

### 2. Query Specificity
Use specific seed queries to avoid irrelevant papers:
- ✅ Good: "RNA-protein binding interface structural analysis"
- ❌ Bad: "RNA protein"

### 3. Avoid Terms
Use avoid terms to filter out irrelevant literature:
- For RNA topics: avoid "DNA-binding", "protein-protein"
- For biophysics: avoid "clinical", "patient"

### 4. Time Windows
Adjust time windows based on topic:
- Fast-moving fields: 6 months
- Established fields: 48-60 months

### 5. Quality Thresholds
Set appropriate quality thresholds for training:
- High threshold (0.8): Only best papers
- Medium threshold (0.5): Good balance
- Low threshold (0.3): More data, lower quality

## Configuration Validation

### Validate Topics

```python
import yaml

def validate_topics(topics_config, domains_config):
    domain_names = set(domains_config["domains"].keys())

    for topic in topics_config["topics"]:
        if topic["domain"] not in domain_names:
            print(f"Error: Domain '{topic['domain']}' not found")
            return False

    return True
```

### Validate Domains

```python
def validate_domains(domains_config):
    required_fields = ["relevance_terms", "gap_categories"]

    for domain_name, domain_config in domains_config["domains"].items():
        for field in required_fields:
            if field not in domain_config:
                print(f"Error: Domain '{domain_name}' missing '{field}'")
                return False

    return True
```

## Configuration Migration

### Schema Versioning

Configuration files should include schema version for future migrations:

```yaml
schema_version: "journal_club_config.v1"
topics:
  ...
```

### Migration Logic

```python
def migrate_config(config):
    version = config.get("schema_version", "v0")

    if version == "v0":
        # Add new fields
        config["schema_version"] = "v1"
        # ... migration logic

    return config
```

## Troubleshooting

### Topics Not Loading
- Check YAML syntax (use YAML validator)
- Verify file path is correct
- Check file permissions

### Domain Filtering Not Working
- Verify domain names match between files
- Check relevance terms coverage
- Test with sample papers

### Settings Not Applied
- Check YAML syntax
- Verify environment variables override correctly
- Restart pipeline after changes

### Training Not Triggering
- Verify `training.enabled: true`
- Check `min_papers_threshold`
- Verify paper count in memory
