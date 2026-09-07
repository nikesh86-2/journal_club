# Literature Memory (SQLite)

`core/literature_memory.py` is the persistence layer: a SQLite database in
WAL mode. It replaced the single-JSON-file design; the public API is
unchanged so callers (streaming, analysis, web) work transparently.

## Database

Default path: `cache/journal_club_memory.db`
(override with `JOURNAL_CLUB_LITERATURE_MEMORY_PATH`).

### Schema

```sql
papers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL, title_norm TEXT NOT NULL,
    abstract TEXT, year TEXT, publication_date TEXT,
    doi TEXT, doi_norm TEXT, pmid TEXT, url TEXT, source TEXT,
    authors TEXT,              -- JSON list
    citation_count INTEGER,
    topic_name TEXT, domain TEXT,
    schema_version TEXT, timestamp TEXT,
    summary TEXT, critique TEXT,
    gap_analysis TEXT,        -- JSON
    quality_scores TEXT       -- JSON
)
metadata(key TEXT PRIMARY KEY, value TEXT)
```

Indexes on `doi_norm`, `title_norm`, `topic_name`, `domain`.

### Concurrency

- `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`.
- One connection per instance, guarded by a re-entrant lock; all mutating
  operations commit per call.
- Multiple processes (web app, streaming, analysis) can safely share the
  database; WAL allows concurrent readers with a single writer.

## API

| Method | Purpose |
|--------|---------|
| `ingest_paper(paper, topic_name, domain)` | Insert a paper (returns the stored record) |
| `update_paper_analysis(key, summary, critique, gap_analysis, quality_scores)` | Update analysis fields, matching raw or normalized DOI/PMID/title |
| `filter_papers(topic, domain, year, doi, pmid, limit, offset)` | Ordered, paginated filtering |
| `get_all_papers()` | All papers |
| `get_papers_by_topic(name, limit)` | Papers for a topic |
| `get_paper_by_key(key)` | Single paper by DOI/PMID/title |
| `search_papers(query, topic, domain, limit, sort_by, order)` | Substring search with title/abstract scoring |
| `get_statistics()` | Live counts: total, analyzed, by topic, by domain |
| `summary()` | Human-readable summary string |
| `paper_exists(paper)` | Dedup check via normalized DOI/PMID/title |
| `deduplicate_papers()` | Remove duplicate DOIs, merging analyses into the kept copy |
| `remove_papers_by_doi(dois)` | Delete papers by DOI |
| `scrub_thinking_traces()` | Remove `<think>` blocks from summaries/critiques |
| `set_metadata(key, value)` / `get_metadata(key)` | Arbitrary key-value store (`training_version`, `last_training_date`, …) |
| `save()` | Commit + prune invalid rows (API compatibility) |
| `memory` (property) | Legacy dict snapshot `{"papers": [...], "statistics": {...}}` |

## Migration from legacy JSON

On first use, if a legacy JSON file exists next to the database path (or a
`*.json` path is passed explicitly), its papers are imported in a single
transaction and the JSON is renamed with a `.migrated` suffix. Empty or
corrupt JSON files are skipped with a warning.

## Notes

- Timestamps are real ingest times (UTC ISO-8601).
- `citation_count` is populated from Semantic Scholar / Europe PMC at ingest
  and can be backfilled with `scripts/backfill_citations.py`.
- The `memory` property builds a full snapshot on each access; prefer the
  query methods in new code.
