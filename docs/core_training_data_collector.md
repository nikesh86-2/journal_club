# Training Data Collector Module

**File**: `core/training_data_collector.py`

## Overview

The training data collector converts analyzed papers into instruction-tuning format for LoRA fine-tuning. It creates multiple types of training examples from paper metadata, analysis results, and recommendations.

## Training Example Types

### 1. Summarization
**Input:** Paper title and abstract
**Output:** Paper summary
**Instruction:** "Summarize the following research paper in 2-3 sentences, focusing on the main research question, methods, and key findings."

### 2. Gap Analysis
**Input:** Paper title and abstract
**Output:** Gap analysis JSON (methodology, controls, statistics, reproducibility)
**Instruction:** "Analyze the following research paper for gaps in methodology, controls, statistics, and reproducibility. Identify specific issues in each category."

### 3. Critique Generation
**Input:** Paper title, abstract, and gap analysis
**Output:** Structured critique
**Instruction:** "Generate a constructive critique of the research paper, highlighting strengths, discussing identified gaps, and suggesting improvements."

### 4. Quality Scoring
**Input:** Paper title, abstract, and gap analysis
**Output:** Quality scores JSON
**Instruction:** "Score the paper on methodology rigor, statistical power, reproducibility, and control quality. Provide scores from 0 to 1 for each metric."

### 5. Recommendations
**Input:** Paper title and abstract
**Output:** Recommendations JSON (foundational, conflicting, related)
**Instruction:** "Recommend foundational papers, conflicting papers, and related reading for the given research paper."

### 6. QA Pairs
**Input:** Paper title, abstract, and question
**Output:** Answer
**Instruction:** "Answer the question based on the research paper."

## Key Classes

### `TrainingDataCollector`

Main class for collecting training data from analyzed papers.

### `__init__(output_dir)`

Initialize the collector.

**Parameters:**
- `output_dir`: Directory to save JSONL files (default: `training/journal_club_data`)

**Behavior:**
- Creates output directory if it doesn't exist
- Generates session ID for this collection run

**Example:**
```python
from core.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector(output_dir="training/journal_club_data")
```

## Key Methods

### `collect_summarization(paper)`

Create a summarization training example.

**Parameters:**
- `paper`: Paper dict with `title`, `abstract`, and `summary`

**Returns:**
- Training example dict or `None`

**Output Format:**
```json
{
  "instruction": "Summarize the following research paper in 2-3 sentences...",
  "input": "Title: {title}\n\nAbstract: {abstract}",
  "output": "{summary}",
  "metadata": {
    "type": "summarization",
    "session_id": "{timestamp}",
    "timestamp": "ISO-8601",
    "topic": "{topic}",
    "domain": "{domain}",
    "paper_doi": "{doi}",
    "quality_score": 0.85
  }
}
```

**Example:**
```python
example = collector.collect_summarization(paper)
```

### `collect_gap_analysis(paper)`

Create a gap analysis training example.

**Parameters:**
- `paper`: Paper dict with `title`, `abstract`, and `gap_analysis`

**Returns:**
- Training example dict or `None`

**Output Format:**
```json
{
  "instruction": "Analyze the following research paper for gaps...",
  "input": "Title: {title}\n\nAbstract: {abstract}",
  "output": "{\"methodology\": [...], \"controls\": [...], ...}",
  "metadata": {
    "type": "gap_analysis",
    "session_id": "{timestamp}",
    "timestamp": "ISO-8601",
    "topic": "{topic}",
    "domain": "{domain}",
    "paper_doi": "{doi}"
  }
}
```

### `collect_critique(paper)`

Create a critique generation training example.

**Parameters:**
- `paper`: Paper dict with `title`, `abstract`, `critique`, and `gap_analysis`

**Returns:**
- Training example dict or `None`

**Output Format:**
```json
{
  "instruction": "Generate a constructive critique of the research paper...",
  "input": "Title: {title}\n\nAbstract: {abstract}\n\nIdentified gaps:...",
  "output": "{critique}",
  "metadata": {
    "type": "critique",
    "session_id": "{timestamp}",
    "timestamp": "ISO-8601",
    "topic": "{topic}",
    "domain": "{domain}",
    "paper_doi": "{doi}"
  }
}
```

### `collect_quality_scoring(paper)`

Create a quality scoring training example.

**Parameters:**
- `paper`: Paper dict with `title`, `abstract`, `quality_scores`, and `gap_analysis`

**Returns:**
- Training example dict or `None`

**Output Format:**
```json
{
  "instruction": "Score the paper on methodology rigor, statistical power...",
  "input": "Title: {title}\n\nAbstract: {abstract}\n\nIdentified gaps:...",
  "output": "{\"methodology_rigor\": 0.8, \"statistical_power\": 0.6, ...}",
  "metadata": {
    "type": "quality_scoring",
    "session_id": "{timestamp}",
    "timestamp": "ISO-8601",
    "topic": "{topic}",
    "domain": "{domain}",
    "paper_doi": "{doi}"
  }
}
```

### `collect_recommendations(paper, recommendations)`

Create a recommendation training example.

**Parameters:**
- `paper`: Paper dict with `title` and `abstract`
- `recommendations`: Recommendations dict from recommendation engine

**Returns:**
- Training example dict or `None`

**Output Format:**
```json
{
  "instruction": "Recommend foundational papers, conflicting papers, and related reading...",
  "input": "Title: {title}\n\nAbstract: {abstract}",
  "output": "{\"foundational\": [...], \"conflicting\": [...], \"related\": [...]}",
  "metadata": {
    "type": "recommendations",
    "session_id": "{timestamp}",
    "timestamp": "ISO-8601",
    "topic": "{topic}",
    "domain": "{domain}",
    "paper_doi": "{doi}"
  }
}
```

### `collect_qa_pair(paper, question, answer)`

Create a QA pair training example.

**Parameters:**
- `paper`: Paper dict
- `question`: Question string
- `answer**: Answer string

**Returns:**
- Training example dict or `None`

**Output Format:**
```json
{
  "instruction": "Answer the question based on the research paper.",
  "input": "Title: {title}\n\nAbstract: {abstract}\n\nQuestion: {question}",
  "output": "{answer}",
  "metadata": {
    "type": "qa_pair",
    "session_id": "{timestamp}",
    "timestamp": "ISO-8601",
    "topic": "{topic}",
    "domain": "{domain}",
    "paper_doi": "{doi}"
  }
}
```

### `collect_all(paper, recommendations)`

Collect all training examples for a paper.

**Parameters:**
- `paper`: Paper dict with analysis results
- `recommendations`: Recommendations dict (optional)

**Returns:**
- List of training example dicts

**Behavior:**
- Calls each collect method if the required data is present
- Returns all successfully created examples
- Logs number of examples collected

**Example:**
```python
examples = collector.collect_all(paper, recommendations)
print(f"Collected {len(examples)} examples")
```

### `collect_batch(papers, recommendations_map)`

Collect training examples for a batch of papers.

**Parameters:**
- `papers`: List of paper dicts
- `recommendations_map`: Dict mapping paper keys to recommendations (optional)

**Returns:**
- Total number of examples collected

**Behavior:**
- Processes each paper in the batch
- Calls `collect_all` for each paper
- Returns total count
- Logs progress

**Example:**
```python
total = collector.collect_batch(papers, recommendations_map)
print(f"Collected {total} total examples")
```

## Module Functions

### `collect_training_data_from_memory(memory_path, output_dir, min_quality_score)`

Collect training data from literature memory.

**Parameters:**
- `memory_path`: Path to literature memory JSON
- `output_dir`: Directory to save JSONL files
- `min_quality_score`: Minimum quality score for inclusion (default: 0.5)

**Returns:**
- Total number of examples collected

**Behavior:**
- Loads literature memory
- Iterates through topics
- Filters papers by quality score
- Collects all example types
- Returns total count

**Example:**
```python
from core.training_data_collector import collect_training_data_from_memory

count = collect_training_data_from_memory(
    memory_path="literature_memory.json",
    output_dir="training/journal_club_data",
    min_quality_score=0.5
)
print(f"Collected {count} examples")
```

## File Organization

### Output Files

Training examples are saved to separate JSONL files by type:

```
training/journal_club_data/
├── summarization_{session_id}.jsonl
├── gap_analysis_{session_id}.jsonl
├── critique_{session_id}.jsonl
├── quality_scoring_{session_id}.jsonl
├── recommendations_{session_id}.jsonl
└── qa_pair_{session_id}.jsonl
```

### Session ID

Each collection run has a unique session ID:
- Format: `YYYYMMDD_HHMMSS`
- Used to group examples by collection run
- Stored in metadata for traceability

## Usage Patterns

### Collect from Memory
```python
from core.training_data_collector import collect_training_data_from_memory

count = collect_training_data_from_memory(
    memory_path="literature_memory.json",
    min_quality_score=0.5
)
```

### Collect Single Paper
```python
from core.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector()
examples = collector.collect_all(paper, recommendations)
```

### Collect Batch
```python
from core.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector()
total = collector.collect_batch(papers, recommendations_map)
```

### Custom QA Pairs
```python
from core.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector()
collector.collect_qa_pair(
    paper,
    question="What are the main findings?",
    answer="The main findings are..."
)
```

## Metadata Fields

Each training example includes metadata:

| Field | Description |
|-------|-------------|
| `type` | Example type (summarization, gap_analysis, etc.) |
| `session_id` | Collection session timestamp |
| `timestamp` | ISO-8601 timestamp |
| `topic` | Topic name if available |
| `domain` | Domain if available |
| `paper_doi` | Paper DOI if available |
| `quality_score` | Overall quality score if available |

## Quality Filtering

Only papers with quality score ≥ `min_quality_score` are included in training data. This ensures:
- High-quality training examples
- Better model performance
- Reduced noise in training data

## Error Handling

- Missing fields in paper dict are skipped
- Papers without required data don't generate examples
- File I/O errors are caught and logged
- Invalid JSON is handled gracefully

## Performance Considerations

- Each paper generates up to 6 examples
- File I/O occurs for each example
- Consider batching for large datasets
- Memory usage scales with number of papers

## Troubleshooting

### No examples collected
- Check papers have analysis results
- Verify quality scores meet threshold
- Check output directory permissions

### Examples missing fields
- Verify paper has required fields (title, abstract, summary, etc.)
- Check that analysis was completed
- Verify recommendations were generated

### File write errors
- Check output directory exists
- Verify write permissions
- Check disk space
