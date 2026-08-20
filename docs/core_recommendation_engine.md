# Recommendation Engine Module

**File**: `core/recommendation_engine.py`

## Overview

The recommendation engine identifies foundational papers, detects conflicting papers, recommends related reading, and clusters papers by theme. It uses both rule-based methods and LLM-assisted analysis to provide comprehensive paper recommendations.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_FAISS_INDEX_PATH` | `cache/faiss_index` | Path to FAISS index |
| `JOURNAL_CLUB_USE_FINETUNED` | `0` | Use fine-tuned model (1 = yes) |
| `JOURNAL_CLUB_FINETUNED_MODEL_PATH` | `training/journal_club_merged_model` | Path to fine-tuned model |
| `JOURNAL_CLUB_FALLBACK_TO_BASE` | `1` | Fall back to base model if fine-tuned fails |

### Foundational Paper Criteria

| Criterion | Value | Description |
|----------|-------|-------------|
| `FOUNDATIONAL_MIN_CITATIONS` | 50 | Minimum citations to be considered |
| `FOUNDATIONAL_MIN_YEARS_OLD` | 3 | Minimum age in years |
| `FOUNDATIONAL_CITATION_THRESHOLD` | 100 | High citation threshold |

## Key Functions

### `get_llm_client(use_finetuned=None)`

Get LLM client with fine-tuned model support.

**Parameters:**
- `use_finetuned`: Override for using fine-tuned model

**Returns:**
- LLM client or `None`

**Behavior:**
- Loads fine-tuned model if enabled and available
- Falls back to VLAB2's LLM
- Falls back to ChatOpenAI
- Returns `None` if no LLM available

### `is_foundational_paper(paper)`

Determine if a paper is foundational (highly-cited, older).

**Parameters:**
- `paper`: Paper dict with `year` and `citation_count`

**Returns:**
- `True` if foundational, `False` otherwise

**Criteria:**
- Must be at least 3 years old
- Must have ≥ 50 citations OR ≥ 100 citations if ≥ 5 years old

**Example:**
```python
paper = {
    "title": "Classic RNA Study",
    "year": "2015",
    "citation_count": 150
}
is_foundational = is_foundational_paper(paper)  # True
```

### `find_foundational_papers(papers, limit)`

Find foundational papers from a list.

**Parameters:**
- `papers`: List of paper dicts
- `limit`: Maximum number to return

**Returns:**
- List of foundational papers sorted by citation count (descending)

**Example:**
```python
foundational = find_foundational_papers(all_papers, limit=10)
# Returns top 10 foundational papers by citation count
```

### `detect_conflicting_papers(paper, other_papers, llm_client)`

Detect papers with contradictory conclusions.

**Parameters:**
- `paper`: Target paper dict
- `other_papers`: List of other paper dicts
- `llm_client`: LLM client (optional, will fetch if not provided)

**Returns:**
- List of conflicting paper dicts

**Behavior:**
1. Rule-based detection first (contradiction indicators)
2. LLM verification if conflicting papers found
3. Returns verified conflicting papers

**Rule-Based Indicators:**
- **Contradictions**: "contradicts", "contrary to", "inconsistent with", "disagrees with", "fails to replicate"
- **Agreements**: "confirms", "replicates", "consistent with", "agrees with", "supports"

**Example:**
```python
conflicting = detect_conflicting_papers(target_paper, other_papers)
```

### `_rule_based_conflict_detection(paper, other_papers)`

Rule-based conflict detection.

**Parameters:**
- `paper`: Target paper dict
- `other_papers`: List of other paper dicts

**Returns:**
- List of potentially conflicting papers

**Logic:**
- Looks for contradiction indicators in other papers
- Checks for title/term overlap (≥ 2 words)
- Returns papers with contradictions and overlap

### `_llm_conflict_verification(paper, candidates, llm_client)`

LLM-assisted conflict verification.

**Parameters:**
- `paper`: Target paper dict
- `candidates`: Potentially conflicting papers
- `llm_client`: LLM client

**Returns:**
- List of verified conflicting papers

**Behavior:**
- Asks LLM to compare each candidate with target
- Returns only papers where LLM confirms contradiction
- Includes candidate if LLM fails (conservative)

### `recommend_related_reading(paper, faiss_index_path, top_k)`

Recommend related papers using semantic search.

**Parameters:**
- `paper`: Paper dict
- `faiss_index_path`: Path to FAISS index
- `top_k`: Number of recommendations

**Returns:**
- List of related paper dicts

**Behavior:**
- Builds query from title and abstract
- Uses `cached_semantic_search` to find related papers
- Filters out self by DOI
- Returns top_k results

**Example:**
```python
related = recommend_related_reading(paper, top_k=5)
```

### `recommend_related_from_faiss(paper, faiss_index_path, top_k)`

Recommend related papers using FAISS similarity search.

**Parameters:**
- `paper`: Paper dict
- `faiss_index_path`: Path to FAISS index
- `top_k`: Number of recommendations

**Returns:**
- List of related paper dicts with similarity scores

**Behavior:**
- Loads FAISS index
- Creates query document from paper
- Performs similarity search
- Returns papers with similarity scores

**Example:**
```python
related = recommend_related_from_faiss(paper, top_k=5)
# Returns papers with 'score' field
```

### `cluster_papers_by_theme(papers, n_clusters)`

Cluster papers by theme using keyword-based clustering.

**Parameters:**
- `papers`: List of paper dicts
- `n_clusters`: Number of clusters to create

**Returns:**
- Dict mapping theme names to paper lists

**Behavior:**
- Extracts keywords from titles and abstracts
- Counts keyword frequency across all papers
- Assigns top keywords as theme names
- Assigns papers to themes based on keyword matches
- Renames themes with actual keywords

**Example:**
```python
clusters = cluster_papers_by_theme(papers, n_clusters=5)
# Returns: {"RNA Binding": [papers...], "Structure": [papers...], ...}
```

### `generate_recommendations(paper, all_papers, faiss_index_path, llm_client)`

Generate all recommendations for a paper.

**Parameters:**
- `paper`: Target paper dict
- `all_papers`: List of all papers
- `faiss_index_path`: Path to FAISS index
- `llm_client`: LLM client (optional)

**Returns:**
- Dict with keys:
  - `foundational`: List of foundational papers
  - `conflicting`: List of conflicting papers
  - `related`: List of related papers

**Behavior:**
1. Finds foundational papers from all papers
2. Detects conflicting papers
3. Recommends related reading from FAISS
4. Recommends related reading from semantic search
5. Combines and deduplicates related papers

**Example:**
```python
recommendations = generate_recommendations(paper, all_papers)
foundational = recommendations["foundational"]
conflicting = recommendations["conflicting"]
related = recommendations["related"]
```

### `batch_recommendations(papers, faiss_index_path, llm_client)`

Generate recommendations for a batch of papers.

**Parameters:**
- `papers`: List of paper dicts
- `faiss_index_path`: Path to FAISS index
- `llm_client`: LLM client (optional)

**Returns:**
- Dict mapping paper keys to recommendation dicts

**Behavior:**
- Processes papers sequentially
- Generates recommendations for each
- Handles errors per-paper
- Returns results for all papers

**Example:**
```python
results = batch_recommendations(papers)
for paper_key, recs in results.items():
    print(f"{paper_key}: {len(recs['related'])} related papers")
```

## Usage Patterns

### Basic Recommendations
```python
from core.recommendation_engine import generate_recommendations

recommendations = generate_recommendations(paper, all_papers)
print(f"Foundational: {len(recommendations['foundational'])}")
print(f"Conflicting: {len(recommendations['conflicting'])}")
print(f"Related: {len(recommendations['related'])}")
```

### Foundational Papers Only
```python
from core.recommendation_engine import find_foundational_papers

foundational = find_foundational_papers(all_papers, limit=10)
for paper in foundational:
    print(f"{paper['title']} ({paper['citation_count']} citations)")
```

### Conflict Detection
```python
from core.recommendation_engine import detect_conflicting_papers

conflicting = detect_conflicting_papers(target_paper, other_papers)
for paper in conflicting:
    print(f"Contradicts: {paper['title']}")
```

### Related Reading
```python
from core.recommendation_engine import recommend_related_reading

related = recommend_related_reading(paper, top_k=5)
for paper in related:
    print(f"Related: {paper['title']}")
```

### Theme Clustering
```python
from core.recommendation_engine import cluster_papers_by_theme

clusters = cluster_papers_by_theme(papers, n_clusters=5)
for theme, theme_papers in clusters.items():
    print(f"{theme}: {len(theme_papers)} papers")
```

### Batch Processing
```python
from core.recommendation_engine import batch_recommendations

results = batch_recommendations(papers)
for paper_key, recs in results.items():
    # Update memory with recommendations
    memory.update_recommendations(paper_key, recs)
```

## Error Handling

- FAISS errors return empty lists
- Semantic search errors return empty lists
- LLM errors fall back to rule-based methods
- Missing fields in paper dict are handled with defaults
- Batch processing continues on individual errors

## Performance Considerations

- Semantic search is cached
- FAISS search is O(1) for similarity queries
- LLM calls are synchronous (consider async for large batches)
- Keyword clustering is O(n*m) where n=papers, m=keywords

## Quality Indicators

### Foundational Paper Quality
- High citation count indicates impact
- Older papers with high citations are foundational
- Recent papers with high citations may be emerging

### Conflict Detection Quality
- Rule-based detection is fast but may have false positives
- LLM verification is more accurate but slower
- Consider both for best results

### Related Reading Quality
- Semantic search provides domain-relevant papers
- FAISS provides similar papers based on embeddings
- Combine both for comprehensive recommendations

## Troubleshooting

### No recommendations returned
- Check FAISS index exists
- Verify papers have titles and abstracts
- Check semantic search API key

### Too many false conflicts
- Adjust contradiction indicators
- Increase overlap threshold
- Use LLM verification

### Clustering produces poor themes
- Increase n_clusters
- Adjust keyword extraction
- Consider using embeddings for clustering

### Related papers not relevant
- Check FAISS index quality
- Verify semantic search queries
- Adjust top_k parameter
