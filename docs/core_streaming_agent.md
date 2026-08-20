# Streaming Agent Module

**File**: `core/streaming_agent.py`

## Overview

The streaming agent continuously ingests recent literature from Semantic Scholar, filters for domain relevance, applies time-based filtering, and stores papers in the literature memory. It runs in background threads to avoid blocking the main pipeline.

## Key Concepts

### Background Streaming
- Each topic/domain combination runs in a separate thread
- Threads are daemon threads (exit when main process exits)
- Streaming can be stopped per-topic or globally

### Deduplication
- Papers are deduplicated by DOI, normalized title, or abstract prefix
- Deduplication cache is in-memory (shared across threads)
- FAISS index also provides deduplication at storage layer

### Filtering Pipeline
1. **Time Filtering**: Papers older than time window are excluded
2. **Domain Relevance**: Papers must match domain-specific terms
3. **Avoid Terms**: Papers with avoid terms are excluded

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_FAISS_INDEX_PATH` | `cache/faiss_index` | Path to FAISS index |
| `JOURNAL_CLUB_STREAM_INTERVAL` | `30` | Seconds between fetch cycles |
| `JOURNAL_CLUB_STREAM_BATCH_SIZE` | `20` | Papers per fetch |
| `JOURNAL_CLUB_STREAM_MAX_IDLE` | `0` | Max idle cycles before stop (0 = infinite) |
| `JOURNAL_CLUB_STREAM_MAX_CYCLES` | `0` | Max total cycles (0 = infinite) |
| `JOURNAL_CLUB_DEDUP_ABSTRACT_PREFIX_LEN` | `500` | Characters for abstract-based dedup |
| `JOURNAL_CLUB_TIME_WINDOW_MONTHS` | `12` | Time window for recent papers |

### YAML Configuration

**domains.yaml**: Domain-specific relevance terms
```yaml
domains:
  biophysics:
    relevance_terms: ["molecular dynamics", "docking", "binding affinity"]
    gap_categories: ["methodology", "controls", "statistics", "reproducibility"]
```

**topics.yaml**: Topic-specific terms
```yaml
topics:
  - name: "RNA-Protein Interactions"
    domain_terms:
      target_classes: ["rna-binding", "ribonucleoprotein"]
      motif_terms: ["stem-loop", "hairpin"]
      avoid_terms: ["dna-binding"]
```

## Key Functions

### `start_streaming(topic_name, domain, queries, time_window_months, memory)`

Start streaming literature ingestion for a topic.

**Parameters:**
- `topic_name`: Topic name for categorization
- `domain`: Domain name for filtering
- `queries`: List of search queries
- `time_window_months`: Time window for filtering
- `memory`: JournalClubMemory instance

**Returns:**
- `True` if started, `False` if already active

**Behavior:**
- Creates a new thread for the topic
- Thread runs `stream_papers` function
- Registers thread in active threads registry
- Returns immediately (non-blocking)

**Example:**
```python
from core.streaming_agent import start_streaming
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
started = start_streaming(
    topic_name="RNA-Protein Interactions",
    domain="biophysics",
    queries=["RNA protein binding", "RNA-protein complex"],
    time_window_months=12,
    memory=memory
)
```

### `stream_papers(topic_name, domain, queries, stop_event, interval, batch_size, time_window_months, memory)`

Main streaming loop (runs in background thread).

**Parameters:**
- `topic_name`: Topic name
- `domain`: Domain name
- `queries`: Search queries
- `stop_event`: Threading event to signal stop
- `interval`: Seconds between fetch cycles
- `batch_size`: Papers per fetch
- `time_window_months`: Time window for filtering
- `memory`: JournalClubMemory instance

**Behavior:**
1. Fetches papers via `cached_semantic_search` for each query
2. Deduplicates papers by DOI/title
3. Filters by time window
4. Filters by domain relevance
5. Ingests into memory
6. Appends to FAISS index
7. Logs statistics
8. Sleeps for interval
9. Repeats until stop_event is set or max cycles/idle reached

**Example:**
```python
import threading
stop_event = threading.Event()

stream_papers(
    topic_name="RNA-Protein Interactions",
    domain="biophysics",
    queries=["RNA protein binding"],
    stop_event=stop_event,
    interval=30,
    batch_size=20,
    time_window_months=12,
    memory=memory
)
```

### `stop_streaming(topic_name, domain)`

Stop streaming for a specific topic.

**Parameters:**
- `topic_name`: Topic name
- `domain`: Domain name

**Returns:**
- `True` if stopped, `False` if not active

**Example:**
```python
stopped = stop_streaming("RNA-Protein Interactions", "biophysics")
```

### `stop_all_streaming()`

Stop all active streaming threads.

**Returns:**
- Number of threads stopped

**Example:**
```python
count = stop_all_streaming()
print(f"Stopped {count} streams")
```

### `active_streams()`

Get list of active streaming topics.

**Returns:**
- List of strings in format `"topic_name::domain"`

**Example:**
```python
streams = active_streams()
for stream in streams:
    print(f"Active: {stream}")
```

### `start_all_topics(memory)`

Start streaming for all configured topics.

**Parameters:**
- `memory`: JournalClubMemory instance

**Returns:**
- Number of topics started

**Behavior:**
- Loads topics from `config/topics.yaml`
- Starts streaming for each topic with its queries
- Logs total started

**Example:**
```python
from core.streaming_agent import start_all_topics
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
started = start_all_topics(memory)
print(f"Started {started} topic streams")
```

## Filtering Functions

### `is_domain_relevant(text, domain, topic_terms)`

Check if paper text is relevant to domain/topic.

**Parameters:**
- `text`: Paper text (title + abstract)
- `domain`: Domain name
- `topic_terms`: Topic-specific terms dict

**Returns:**
- `True` if relevant, `False` otherwise

**Behavior:**
- Loads domain relevance terms from `domains.yaml`
- Checks for presence of relevance terms
- Checks for absence of avoid terms
- Returns `True` only if relevance terms present and avoid terms absent

**Example:**
```python
relevant = is_domain_relevant(
    text="RNA-protein binding mechanisms...",
    domain="biophysics",
    topic_terms={"target_classes": ["rna-binding"], "avoid_terms": ["dna-binding"]}
)
```

### `_is_within_time_window(paper, months)`

Check if paper is within time window.

**Parameters:**
- `paper`: Paper dict
- `months`: Time window in months

**Returns:**
- `True` if within window, `False` otherwise

**Behavior:**
- Extracts publication date or year
- Parses date string
- Compares to current date minus threshold
- Returns `True` if no date (include by default)

## Normalization Functions

### `_normalise_text(text)`

Normalize text for comparison.

**Behavior:**
- Converts to lowercase
- Removes URLs
- Removes non-alphanumeric characters
- Collapses whitespace

### `_normalise_title(title)`

Normalize title for comparison.

**Behavior:**
- Same as `_normalise_text`

### `_stable_hash(text)`

Generate stable hash for deduplication.

**Behavior:**
- SHA1 hash of text
- Used for abstract-based deduplication

### `_paper_key(p)`

Generate unique key for paper deduplication.

**Behavior:**
- Returns `doi::{doi}` if DOI present
- Returns `pmid::{pmid}` if PMID present
- Returns `title::{normalized_title}::year::{year}::sig::{hash}` otherwise
- Returns `source::{source}::year::{year}::sig::{hash}` as fallback

## FAISS Integration

### `append_to_faiss(docs)`

Append documents to FAISS index.

**Parameters:**
- `docs`: List of LangChain Document objects

**Returns:**
- Number of documents added

**Behavior:**
- Loads existing FAISS index
- Deduplicates by document content prefix
- Adds unique documents
- Saves updated index

**Example:**
```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Paper title\n\nPaper abstract",
        metadata={"title": "Paper title", "source": "journal_club"}
    )
]
added = append_to_faiss(docs)
```

## Usage Patterns

### Basic Streaming
```python
from core.streaming_agent import start_streaming
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()

# Start streaming for a single topic
start_streaming(
    topic_name="My Topic",
    domain="biophysics",
    queries=["my query"],
    memory=memory
)

# Check active streams
print(active_streams())

# Stop streaming
stop_streaming("My Topic", "biophysics")
```

### Multi-Topic Streaming
```python
from core.streaming_agent import start_all_topics
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()

# Start all configured topics
started = start_all_topics(memory)
print(f"Started {started} streams")

# Later, stop all
stopped = stop_all_streaming()
print(f"Stopped {stopped} streams")
```

### Custom Streaming Loop
```python
import threading
from core.streaming_agent import stream_papers
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
stop_event = threading.Event()

# Start custom stream
thread = threading.Thread(
    target=stream_papers,
    args=("My Topic", "biophysics", ["query"], stop_event),
    daemon=True
)
thread.start()

# Stop after some time
import time
time.sleep(300)
stop_event.set()
thread.join()
```

## Error Handling

- Semantic search errors are caught and logged
- FAISS errors are caught and logged
- Invalid papers are skipped
- Thread errors are caught and logged
- Missing configuration is handled gracefully

## Performance Considerations

- Each stream runs in separate thread
- FAISS operations are thread-safe with locking
- Deduplication cache is in-memory (O(1) lookup)
- Network latency from Semantic Scholar API
- FAISS index size affects append performance

## Monitoring

### Logging
Streaming agent logs at INFO level:
- Stream start/stop events
- Papers added per cycle
- Idle cycles
- Error conditions

### Statistics
Track:
- Active streams count
- Papers added per topic
- Idle cycles per topic
- Total cycles per topic

## Troubleshooting

### No papers being ingested
- Check Semantic Scholar API key
- Verify queries are returning results
- Check time window (may be too restrictive)
- Check domain relevance terms

### High memory usage
- Reduce `STREAM_BATCH_SIZE`
- Reduce `STREAM_MAX_IDLE` to stop idle streams
- Clear FAISS index periodically

### Slow streaming
- Increase `STREAM_INTERVAL` to reduce API calls
- Reduce number of queries per topic
- Check network connectivity

### Duplicate papers
- Check deduplication cache size
- Verify DOI/PMID extraction
- Check FAISS deduplication
