# Literature Memory Module

**File**: `core/literature_memory.py`

## Overview

The `JournalClubMemory` class provides persistent storage for analyzed papers, gap analysis results, quality scores, and recommendations. It serves as the central data store for the Journal Club pipeline.

## Schema

### Memory Structure

```python
{
    "schema_version": "journal_club_memory.v1",
    "papers": [...],                    # List of paper records
    "topic_records": [...],             # Topic-level metadata
    "domain_records": [...],             # Domain-level metadata
    "training_version": 0,               # Training version tracker
    "last_training_date": "ISO-8601"    # Last training timestamp
}
```

### Paper Record Schema

```python
{
    "schema_version": "journal_club_paper.v1",
    "timestamp": "ISO-8601",
    "title": str,
    "abstract": str,
    "year": str | int,
    "publication_date": str | None,
    "doi": str | None,
    "pmid": str | None,
    "url": str | None,
    "source": str,
    
    # Journal club fields
    "topic_name": str | None,
    "domain": str | None,
    
    # Analysis results
    "summary": str | None,
    "critique": str | None,
    "gap_analysis": {
        "methodology": [str],
        "controls": [str],
        "statistics": [str],
        "reproducibility": [str]
    },
    "quality_scores": {
        "methodology_rigor": float,
        "statistical_power": float,
        "reproducibility_score": float,
        "control_quality": float,
        "overall_quality": float
    },
    
    # Recommendations
    "recommendation_type": str | None,  # "foundational", "conflicting", "standard"
    "related_papers": [dict]
}
```

## Key Functions

### `__init__(path: str | None = None)`

Initialize the memory instance.

**Parameters:**
- `path`: Path to the JSON file (default: from `JOURNAL_CLUB_LITERATURE_MEMORY_PATH` env var or `literature_memory.json`)

**Behavior:**
- Loads existing memory from file if it exists
- Initializes empty memory structure if file doesn't exist
- Sets schema version to `journal_club_memory.v1`

### `ingest_paper(paper, topic_name=None, domain=None, time_window_months=12)`

Ingest a new paper into memory.

**Parameters:**
- `paper`: Paper dict with at least `title` and `abstract`
- `topic_name`: Topic name for categorization
- `domain`: Domain for categorization
- `time_window_months`: Time window for filtering (default: 12)

**Returns:**
- Paper record dict if successful, `None` if filtered out

**Behavior:**
- Extracts metadata (title, abstract, doi, pmid, year, url, source)
- Parses publication date
- Filters by time window (papers older than threshold are excluded)
- Creates paper record with journal club fields initialized to `None`
- Deduplicates based on DOI, PMID, or normalized title
- Saves to file

**Example:**
```python
memory = JournalClubMemory()
paper = {
    "title": "RNA-Protein Binding Mechanisms",
    "abstract": "This study investigates...",
    "year": "2024",
    "doi": "10.1234/example.2024.001"
}
record = memory.ingest_paper(paper, topic_name="RNA-Protein Interactions", domain="biophysics")
```

### `update_paper_analysis(key, summary=None, critique=None, gap_analysis=None, quality_scores=None)`

Update analysis results for a paper.

**Parameters:**
- `key`: Paper identifier (DOI or title)
- `summary`: Paper summary
- `critique`: Structured critique
- `gap_analysis`: Gap analysis dict
- `quality_scores`: Quality scores dict

**Behavior:**
- Finds paper by DOI or title
- Updates specified fields
- Saves to file

**Example:**
```python
memory.update_paper_analysis(
    "10.1234/example.2024.001",
    summary="This paper studies RNA-protein binding...",
    gap_analysis={"methodology": ["Limited sample size"]},
    quality_scores={"overall_quality": 0.75}
)
```

### `get_papers_by_topic(topic_name, limit=None)`

Retrieve papers for a specific topic.

**Parameters:**
- `topic_name`: Topic name
- `limit`: Maximum number of papers to return

**Returns:**
- List of paper dicts

**Example:**
```python
papers = memory.get_papers_by_topic("RNA-Protein Interactions", limit=50)
```

### `get_papers_by_domain(domain, limit=None)`

Retrieve papers for a specific domain.

**Parameters:**
- `domain`: Domain name
- `limit`: Maximum number of papers to return

**Returns:**
- List of paper dicts

**Example:**
```python
papers = memory.get_papers_by_domain("biophysics", limit=100)
```

### `get_paper_by_key(key)`

Retrieve a specific paper by DOI or title.

**Parameters:**
- `key`: DOI or title

**Returns:**
- Paper dict or `None`

**Example:**
```python
paper = memory.get_paper_by_key("10.1234/example.2024.001")
```

### `get_statistics()`

Get overall statistics about the memory.

**Returns:**
```python
{
    "total_papers": int,
    "analyzed_papers": int,
    "by_topic": {topic_name: count},
    "by_domain": {domain: count}
}
```

**Example:**
```python
stats = memory.get_statistics()
print(f"Total papers: {stats['total_papers']}")
print(f"Analyzed: {stats['analyzed_papers']}")
```

### `save()`

Save memory to file.

**Behavior:**
- Compacts memory (deduplicates papers)
- Updates schema version
- Writes to JSON file

### `_load()`

Load memory from file.

**Behavior:**
- Reads JSON file if it exists
- Updates memory structure
- Handles corrupt files gracefully

### `_compact()`

Deduplicate papers and bound memory growth.

**Behavior:**
- Removes duplicate papers based on DOI, PMID, or title
- Keeps only most recent 50 topic records
- Keeps only most recent 50 domain records

## Utility Functions

### `_clean_text(text: str) -> str`

Normalize text for comparison.

**Behavior:**
- Converts to lowercase
- Removes URLs
- Removes non-alphanumeric characters
- Collapses whitespace

### `_normalise_key_text(text: str) -> str`

Normalize text for use as deduplication key.

**Behavior:**
- Similar to `_clean_text` but preserves more structure

### `_now_iso() -> str`

Get current timestamp in ISO-8601 format.

### `_parse_publication_date(date_str: str | None) -> datetime | None`

Parse various date formats to datetime.

**Supported Formats:**
- `YYYY-MM-DD`
- `YYYY-MM`
- `YYYY`
- `YYYY-MM-DDTHH:MM:SSZ`
- `YYYY-MM-DDTHH:MM:SS.fZ`

**Returns:**
- `datetime` object or `None` if parsing fails

### `_is_within_time_window(pub_date: str | None, months: int) -> bool`

Check if publication date is within time window.

**Behavior:**
- Returns `True` if no date provided (include by default)
- Returns `True` if parsing fails (include by default)
- Returns `True` if date is within `months` of current date
- Returns `False` if date is older than threshold

## Usage Patterns

### Basic Usage
```python
from core.literature_memory import JournalClubMemory

# Initialize
memory = JournalClubMemory("literature_memory.json")

# Ingest papers
for paper in papers:
    memory.ingest_paper(paper, topic_name="My Topic", domain="biophysics")

# Update analysis
memory.update_paper_analysis(
    "10.1234/example",
    summary="Paper summary...",
    gap_analysis={"methodology": ["Issue 1"]},
    quality_scores={"overall_quality": 0.8}
)

# Query papers
papers = memory.get_papers_by_topic("My Topic")

# Get statistics
stats = memory.get_statistics()
```

### Batch Processing
```python
memory = JournalClubMemory()

# Ingest batch
for paper in paper_batch:
    memory.ingest_paper(paper, topic_name="Batch Topic")

# Save once
memory.save()
```

### Filtering
```python
memory = JournalClubMemory()

# Get only analyzed papers
all_papers = memory.get_papers_by_topic("My Topic")
analyzed = [p for p in all_papers if p.get("summary")]

# Get high-quality papers
high_quality = [p for p in all_papers if p.get("quality_scores", {}).get("overall_quality", 0) >= 0.8]
```

## Error Handling

- File I/O errors are caught and logged
- Corrupt JSON files are handled gracefully (memory reinitialized)
- Missing fields in paper dicts are handled with defaults
- Duplicate papers are silently ignored

## Performance Considerations

- Memory is loaded entirely into RAM
- Deduplication is O(n) where n is number of papers
- File I/O occurs on every save operation
- For large datasets (>10k papers), consider database backend

## Migration

Schema version is tracked in memory. Future versions will include migration logic to upgrade old schemas.

Current schema: `journal_club_memory.v1`
