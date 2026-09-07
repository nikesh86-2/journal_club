# Pipeline: Streaming, Analysis, Recommendations, Reports

## Streaming (`core/streaming_agent.py`)

`start_all_topics(memory)` launches one daemon thread per configured topic.
Each cycle:

1. **Fetch** — `cached_semantic_search` (Semantic Scholar, in-memory cache,
   rate-limited) plus `fetch_europepmc_papers` (Europe PMC `SRC:PPR`,
   `FIRST_PDATE desc`, paged up to 500 results when the window is unlimited).
2. **Filter** — time window (`_is_within_time_window`), then
   `is_domain_relevant`: hard-reject `BAD_TERMS` and topic `avoid_terms`,
   then require a topic-term match AND a domain-term match.
3. **Dedup** — skip keys seen this process (`_stream_cache`) and papers
   already in memory (`memory.paper_exists`). A `deduplicate_papers()` pass
   runs at stream startup.
4. **Ingest + analyze** — `ingest_paper` stores the record; `ingest_into_analysis`
   computes summary, gap analysis, and quality scores and persists them by
   the paper's raw DOI/PMID/title.
5. **Index** — new papers are appended to the FAISS index.

Entry points: `start_streaming`, `stop_streaming`, `stop_all_streaming`,
`active_streams`. Stops after `JOURNAL_CLUB_STREAM_MAX_IDLE` idle cycles or
`JOURNAL_CLUB_STREAM_MAX_CYCLES` total cycles when set.

## Analysis (`core/paper_analyzer.py`)

`analyze_paper(paper, domain, related_papers, memory)`:

1. Returns the cached result when available (`cache/analysis/`,
   schema `analysis.v2` — older caches are ignored).
2. `generate_summary` — 2-3 sentence LLM summary.
3. `analyze_gaps` — structured JSON (`methodology`, `controls`,
   `statistics`, `reproducibility`) validated with Pydantic, re-invoked on
   parse failure, rule-based fallback.
4. `generate_critique` — structured peer-review critique.
5. `score_paper_quality` — LLM rubric scoring (0-1 per dimension) with a
   deterministic rule-based fallback; the result carries a
   `scoring_method` label (`llm` or `rule_based`).

`analyze_batch(papers, domain, memory)` runs papers in a thread pool and
**returns results in input order**; failures produce placeholder results
rather than shifting indices.

Batch entry point: `python scripts/run_analysis.py [--all] [--limit N]` —
uses each topic's configured domain (falling back to the paper's stored
domain, then `general`).

### LLM clients

`get_llm_client()` caches one client per key and picks the first available:

1. Fine-tuned merged model (`JOURNAL_CLUB_USE_FINETUNED=1` and model exists)
2. `LlamaServerLLM` — external llama-server (`/v1/chat/completions`), health
   check **raises** on failure so fallback proceeds; requests retried with
   backoff.
3. `GGUFChatLLM` — llama-cpp-python GGUF.
4. Local HuggingFace pipeline (8-bit/CPU-offload fallback ladder).
5. `ChatOpenAI`.

All analysis calls go through `invoke_llm()`, which disables reasoning mode
(`enable_thinking=False`) where supported; `_get_response_text()` strips
`<think>` blocks (including unclosed ones) as a safety net.

## Recommendations (`core/recommendation_engine.py`)

`generate_recommendations(paper, all_papers)` returns:

- **foundational** — `citation_count >= 100`, or `>= 50` and `>= 5` years
  old (requires backfilled citation counts).
- **conflicting** — rule-based contradiction indicators (title overlap +
  negation terms), verified by the LLM when available.
- **related** — FAISS similarity + Semantic Scholar semantic search,
  deduplicated by DOI.

`cluster_papers_by_theme` groups papers by top keywords; `batch_recommendations`
wraps the above for a collection.

## Reports (`core/report_generator.py`)

- `generate_topic_report(topic, memory, output_dir)` — per-topic markdown.
- `generate_paper_report(paper)` — single-paper markdown.
- `generate_summary_report(memory, output_dir)` — pipeline statistics.
- `generate_all_reports(memory)` — all of the above.

Quality sections show the five rubric dimensions plus the `scoring_method`
label. Outputs land in `output/markdown/` and `output/json/`.
