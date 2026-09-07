# Web Interface (`web/app.py`)

Flask app backed by **one shared memory instance** (created lazily, reused
for all requests — no per-request reloads).

## Pages

| Route | Description |
|-------|-------------|
| `/` | Dashboard: stats, topics, search, config, ingest/analysis controls |
| `/topic/<name>` | Topic papers, **paginated** 50/page (`?page=N`) |
| `/paper/<key>` | Paper detail (DOI, PMID, or title key) |

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/stats` | Memory statistics |
| `GET /api/topics` | Topic names |
| `GET /api/papers?topic=&domain=&limit=&offset=` | Paginated paper list |
| `GET /api/papers/<topic>?limit=&offset=` | Paginated papers by topic |
| `GET /api/paper/<key>` | Single paper |
| `POST /api/trigger-analysis` | Run `analyze_batch` over unanalyzed papers |
| `POST /api/refresh-stats` | Recompute statistics |
| `GET /api/search?q=&limit=` | **Semantic search** over the local FAISS index |
| `GET/POST /api/config/sources` | LLM/search settings (writes `settings.yaml`) |
| `POST /api/config/test-source` | Test LLM endpoint + Europe PMC connectivity |
| `GET/POST /api/config/topics` | List / add / update topics |
| `DELETE /api/config/topics/<name>` | Delete a topic |
| `GET/POST /api/config/domains` | List / add domains |
| `POST /api/ingest` | On-demand ingestion for a topic or custom query |
| `GET /export/<topic>/markdown` | Download topic markdown report |
| `GET /export/<topic>/json` | Download topic JSON export |
| `GET /export/summary/markdown` | Download summary report |

## Semantic search

`GET /api/search?q=...` runs the query against the FAISS index
(`core/research_agent_adaptive.search_local_db`) and returns the top-K
documents with title, DOI, year, topic, and a 500-char abstract snippet.
The dashboard has a search box wired to this endpoint. Build/refresh the
index with `python scripts/build_faiss_index.py`.

## Running

```bash
./scripts/run_journal_club.sh web        # or: python web/app.py
```

Port from `JOURNAL_CLUB_WEB_PORT` (default 5000).
