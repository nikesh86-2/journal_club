# Paper Analyzer Module

**File**: `core/paper_analyzer.py`

## Overview

The paper analyzer performs comprehensive analysis of research papers, including summarization, gap analysis (methodology, controls, statistics, reproducibility), quality scoring, and critique generation. It can use either the base LLM or a fine-tuned model for improved domain-specific analysis.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_LLM_MODEL` | `gpt-4` | Base LLM model name |
| `JOURNAL_CLUB_LLM_TEMPERATURE` | `0.3` | LLM temperature |
| `JOURNAL_CLUB_LLM_MAX_TOKENS` | `2000` | Max tokens per response |
| `JOURNAL_CLUB_USE_FINETUNED` | `0` | Use fine-tuned model (1 = yes) |
| `JOURNAL_CLUB_FINETUNED_MODEL_PATH` | `training/journal_club_merged_model` | Path to fine-tuned model |
| `JOURNAL_CLUB_FALLBACK_TO_BASE` | `1` | Fall back to base model if fine-tuned fails |

## Key Functions

### `get_llm_client(use_finetuned=None)`

Get LLM client with fine-tuned model support.

**Parameters:**
- `use_finetuned`: Override for using fine-tuned model (default: from env var)

**Returns:**
- LLM client (ChatOpenAI or HuggingFacePipeline) or `None`

**Behavior:**
1. If `use_finetuned` is `True`:
   - Loads fine-tuned model from `FINETUNED_MODEL_PATH`
   - Uses HuggingFacePipeline for local model
   - Falls back to base model if loading fails (if `FALLBACK_TO_BASE=1`)
2. If fine-tuned not used or fails:
   - Tries VLAB2's `get_llm()`
   - Falls back to ChatOpenAI
3. Returns `None` if no LLM available

**Example:**
```python
from core.paper_analyzer import get_llm_client

# Use fine-tuned model
llm = get_llm_client(use_finetuned=True)

# Use base model
llm = get_llm_client(use_finetuned=False)
```

### `generate_summary(paper)`

Generate a concise summary of the paper.

**Parameters:**
- `paper`: Paper dict with `title` and `abstract`

**Returns:**
- Summary string

**Behavior:**
- If LLM available: Generates 2-3 sentence summary via LLM
- If LLM unavailable: Extracts first 3 sentences from abstract
- Focuses on research question, methods, and conclusions

**Example:**
```python
paper = {
    "title": "RNA-Protein Binding Mechanisms",
    "abstract": "This study investigates the molecular mechanisms..."
}
summary = generate_summary(paper)
```

### `analyze_gaps(paper, domain)`

Analyze gaps in methodology, controls, statistics, and reproducibility.

**Parameters:**
- `paper`: Paper dict with `title` and `abstract`
- `domain`: Domain name for context

**Returns:**
- Dict with keys: `methodology`, `controls`, `statistics`, `reproducibility`
- Each key contains a list of gap descriptions

**Behavior:**
- If LLM available: Uses LLM to identify specific gaps
- If LLM unavailable: Uses rule-based detection
- Returns JSON-structured gap analysis

**Rule-Based Detection:**
- **Methodology**: Detects "preliminary", "pilot", "in vitro only"
- **Controls**: Checks for "control" mentions
- **Statistics**: Checks for "p-value", "sample size", "statistically"
- **Reproducibility**: Checks for "data available", "proprietary", "supplementary"

**Example:**
```python
gaps = analyze_gaps(paper, domain="biophysics")
# Returns:
# {
#     "methodology": ["Preliminary study with limited validation"],
#     "controls": ["No explicit mention of control experiments"],
#     "statistics": ["No statistical analysis reported"],
#     "reproducibility": ["Data availability not mentioned"]
# }
```

### `generate_critique(paper, gap_analysis, related_papers)`

Generate a structured critique of the paper.

**Parameters:**
- `paper`: Paper dict
- `gap_analysis`: Gap analysis dict from `analyze_gaps`
- `related_papers`: List of related papers for context (optional)

**Returns:**
- Critique string

**Behavior:**
- If LLM available: Generates 3-4 paragraph critique
- If LLM unavailable: Uses rule-based critique
- Includes summary of contribution, strengths, gaps, improvements, contradictions

**Example:**
```python
critique = generate_critique(paper, gap_analysis, related_papers)
```

### `score_paper_quality(paper, gap_analysis)`

Score paper on quality dimensions.

**Parameters:**
- `paper`: Paper dict
- `gap_analysis`: Gap analysis dict

**Returns:**
- Dict with keys:
  - `methodology_rigor` (0-1)
  - `statistical_power` (0-1)
  - `reproducibility_score` (0-1)
  - `control_quality` (0-1)
  - `overall_quality` (0-1)

**Scoring Logic:**
- Base score: 1.0 minus (gap_count * penalty)
- Penalties: methodology (0.3), statistics (0.4), reproducibility (0.3), controls (0.4)
- Boosts: "randomized" (+0.1), "blinded" (+0.1), "replicate" (+0.1), "sample size" (+0.1)
- Overall: Weighted average (30% methodology, 25% statistics, 25% reproducibility, 20% controls)

**Example:**
```python
scores = score_paper_quality(paper, gap_analysis)
# Returns:
# {
#     "methodology_rigor": 0.8,
#     "statistical_power": 0.6,
#     "reproducibility_score": 0.9,
#     "control_quality": 0.7,
#     "overall_quality": 0.75
# }
```

### `analyze_paper(paper, domain, related_papers)`

Run full analysis pipeline on a paper.

**Parameters:**
- `paper`: Paper dict
- `domain`: Domain name
- `related_papers`: Related papers for context (optional)

**Returns:**
- Dict with keys:
  - `summary`: Summary string
  - `gap_analysis`: Gap analysis dict
  - `critique`: Critique string
  - `quality_scores`: Quality scores dict

**Behavior:**
1. Generates summary
2. Analyzes gaps
3. Generates critique
4. Scores quality
5. Returns combined analysis

**Example:**
```python
analysis = analyze_paper(paper, domain="biophysics")
summary = analysis["summary"]
gaps = analysis["gap_analysis"]
critique = analysis["critique"]
scores = analysis["quality_scores"]
```

### `analyze_batch(papers, domain)`

Analyze a batch of papers.

**Parameters:**
- `papers`: List of paper dicts
- `domain`: Domain name

**Returns:**
- List of dicts with keys:
  - `paper`: Original paper dict
  - `analysis`: Analysis dict or `None`
  - `error`: Error string (if analysis failed)

**Behavior:**
- Processes papers sequentially
- Logs progress
- Handles errors per-paper (doesn't fail entire batch)
- Returns results for all papers

**Example:**
```python
results = analyze_batch(papers, domain="biophysics")
for result in results:
    if result["analysis"]:
        print(f"Analyzed: {result['paper']['title']}")
    else:
        print(f"Failed: {result['error']}")
```

## Rule-Based Functions

### `_rule_based_gap_analysis(text)`

Rule-based gap analysis as fallback.

**Parameters:**
- `text`: Combined title and abstract text

**Returns:**
- Gap analysis dict

**Indicators:**
- **Methodology**: "preliminary", "pilot", "in vitro only"
- **Controls**: Missing "control" mentions
- **Statistics**: Missing "p-value", "sample size", "statistically"
- **Reproducibility**: Missing "data available", "proprietary"

### `_rule_based_critique(paper, gap_analysis)`

Rule-based critique as fallback.

**Parameters:**
- `paper`: Paper dict
- `gap_analysis`: Gap analysis dict

**Returns:**
- Critique string

**Logic:**
- Counts total gaps
- Provides summary based on gap count
- Lists specific gap categories

## Usage Patterns

### Basic Analysis
```python
from core.paper_analyzer import analyze_paper

paper = {
    "title": "RNA-Protein Binding",
    "abstract": "This study..."
}

analysis = analyze_paper(paper, domain="biophysics")
print(analysis["summary"])
print(analysis["quality_scores"]["overall_quality"])
```

### Batch Analysis
```python
from core.paper_analyzer import analyze_batch

papers = [paper1, paper2, paper3]
results = analyze_batch(papers, domain="biophysics")

for result in results:
    if result["analysis"]:
        # Update memory with analysis
        memory.update_paper_analysis(
            result["paper"]["doi"],
            summary=result["analysis"]["summary"],
            gap_analysis=result["analysis"]["gap_analysis"],
            quality_scores=result["analysis"]["quality_scores"]
        )
```

### Using Fine-Tuned Model
```python
import os
os.environ["JOURNAL_CLUB_USE_FINETUNED"] = "1"

from core.paper_analyzer import analyze_paper

analysis = analyze_paper(paper, domain="biophysics")
# Will use fine-tuned model if available
```

### Individual Components
```python
from core.paper_analyzer import generate_summary, analyze_gaps, score_paper_quality

# Just summary
summary = generate_summary(paper)

# Just gap analysis
gaps = analyze_gaps(paper, domain="biophysics")

# Just quality scores
scores = score_paper_quality(paper, gaps)
```

## Error Handling

- LLM errors fall back to rule-based methods
- Missing fields in paper dict are handled with defaults
- Batch analysis continues on individual errors
- Fine-tuned model loading errors fall back to base model

## Performance Considerations

- LLM calls are synchronous (consider async for large batches)
- Rule-based methods are much faster than LLM
- Fine-tuned model loading is one-time cost per process
- Batch analysis processes sequentially (parallelize for speed)

## Quality Indicators

### Quality Score Interpretation
- **0.8-1.0**: High quality (green)
- **0.6-0.8**: Medium quality (yellow)
- **0.0-0.6**: Low quality (red)

### Gap Categories
- **Methodology**: Study design, experimental approach
- **Controls**: Appropriate control experiments
- **Statistics**: Sample size, statistical tests, p-values
- **Reproducibility**: Data availability, method detail, proprietary methods

## Troubleshooting

### Analysis returns empty results
- Check LLM API key
- Verify paper has title and abstract
- Check domain configuration

### Quality scores seem incorrect
- Review gap analysis results
- Check rule-based indicators
- Adjust penalty weights in code

### Fine-tuned model not loading
- Check `FINETUNED_MODEL_PATH` is correct
- Verify model files exist
- Check `FALLBACK_TO_BASE` setting

### Batch analysis is slow
- Reduce batch size
- Use rule-based methods only
- Consider parallel processing
