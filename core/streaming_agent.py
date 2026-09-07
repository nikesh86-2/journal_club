"""
streaming_agent.py

Background literature ingestion for Journal Club.

Responsibilities:
  - Start one background stream per topic/domain profile
  - Fetch papers using semantic search
  - Filter for domain-specific relevance using configurable terms
  - Filter by publication date (configurable time window)
  - Deduplicate papers by DOI, normalized title
  - Append genuinely new documents to FAISS
  - Ingest papers into JournalClubMemory

Important behaviour:
  - "No new papers" is logged at DEBUG, not INFO
  - FAISS path is configurable
  - Domain terms are loaded from YAML configuration
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import yaml

from . import config
from .research_agent_adaptive import (
    CachedSentenceTransformerEmbeddings,
    cached_semantic_search,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .literature_memory import JournalClubMemory
from .paper_analyzer import cleanup_llm_clients

log = logging.getLogger("journal_club.streaming")

# Define quiet time handler
_quiet_time = datetime.now()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_stream_cache: Set[str] = set()
_active_queries: Dict[str, threading.Event] = {}
_active_threads: Dict[str, threading.Thread] = {}
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = config.DEFAULT_FAISS_INDEX_PATH

STREAM_INTERVAL = int(os.getenv("JOURNAL_CLUB_STREAM_INTERVAL", "30"))
STREAM_BATCH_SIZE = int(os.getenv("JOURNAL_CLUB_STREAM_BATCH_SIZE", "20"))
STREAM_MAX_IDLE = int(os.getenv("JOURNAL_CLUB_STREAM_MAX_IDLE", "0"))
STREAM_MAX_CYCLES = int(os.getenv("JOURNAL_CLUB_STREAM_MAX_CYCLES", "0"))
DEDUP_ABSTRACT_PREFIX_LEN = int(os.getenv("JOURNAL_CLUB_DEDUP_ABSTRACT_PREFIX_LEN", "500"))

DEFAULT_TIME_WINDOW_MONTHS = int(os.getenv("JOURNAL_CLUB_TIME_WINDOW_MONTHS", "0"))

# Whether to query Semantic Scholar during streaming ingestion. Disable to
# rely solely on Europe PMC (useful when the Semantic Scholar API is slow or
# rate-limited).
ENABLE_SEMANTIC_SCHOLAR = os.getenv("JOURNAL_CLUB_ENABLE_SEMANTIC_SCHOLAR", "1") == "1"

# ---------------------------------------------------------------------------
# Hard-reject terms — papers matching any of these are never relevant
# ---------------------------------------------------------------------------

BAD_TERMS = [
    "multi-agent reinforcement learning",
    "multi-agent llm",
    "large language model",
    "llm planning",
    "jailbreak",
    "manufacturing systems",
    "pose graph",
    "clinical trial multi-agent",
    "robot",
    "slam",
    "phosphate glass",
    "plantaricin",
    "anti-cancer",
    "anticancer",
    "machine learning pipeline",
    "deep learning framework",
    "neural network architecture",
    "natural language processing",
    "computer vision",
    "autonomous driving",
    "knowledge graph",
    "plate composition",
    "lab automation",
]

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------

def load_domain_config(config_path: str | None = None) -> dict:
    """Load domain configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parents[1] / "config" / "domains.yaml"

    if not os.path.exists(config_path):
        log.warning("Domain config not found at %s, using empty config", config_path)
        return {}

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.warning("Failed to load domain config: %s", e)
        return {}


def load_topic_config(config_path: str | None = None) -> list:
    """Load topic configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parents[1] / "config" / "topics.yaml"

    if not os.path.exists(config_path):
        log.warning("Topic config not found at %s, using empty list", config_path)
        return []

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
            return data.get("topics", [])
    except Exception as e:
        log.warning("Failed to load topic config: %s", e)
        return []


_domain_config = load_domain_config()
_topic_config = load_topic_config()


def get_domain_terms(domain: str) -> dict:
    """Get relevance terms for a domain."""
    domain_data = _domain_config.get("domains", {}).get(domain, {})
    return {
        "relevance_terms": domain_data.get("relevance_terms", []),
        "gap_categories": domain_data.get("gap_categories", []),
        "quality_metrics": domain_data.get("quality_metrics", []),
    }


def get_topic_config(topic_name: str) -> dict | None:
    """Get configuration for a specific topic."""
    for topic in _topic_config:
        if topic.get("name") == topic_name:
            return topic
    return None


# ---------------------------------------------------------------------------
# Normalisation / dedupe
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalise_title(title: str) -> str:
    return _normalise_text(title)


def _stable_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _paper_key(p: dict) -> str:
    title = _normalise_title(p.get("title") or "")
    doi = _normalise_text(p.get("doi") or p.get("DOI") or "")
    year = str(p.get("year") or "").strip()
    source = _normalise_text(p.get("source") or "semantic_search")

    if doi:
        return f"doi::{doi}"

    abstract = _normalise_text(p.get("abstract") or "")[:DEDUP_ABSTRACT_PREFIX_LEN]
    content_sig = _stable_hash(f"{title}|{year}|{abstract}")

    if title:
        return f"title::{title}|year::{year}|sig::{content_sig}"

    return f"source::{source}|year::{year}|sig::{content_sig}"


# ---------------------------------------------------------------------------
# Time filtering
# ---------------------------------------------------------------------------

def _parse_publication_date(date_str: str | None) -> datetime | None:
    """Parse various date formats to datetime."""
    if not date_str:
        return None

    date_str = str(date_str).strip()

    formats = [
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def _is_within_time_window(paper: dict, months: int) -> bool:
    """Check if paper is within time window. If months <= 0, include all papers."""
    # Allow unlimited historical papers
    if months <= 0:
        return True

    pub_date = paper.get("publication_date") or paper.get("year")

    if not pub_date:
        return True  # Include if no date

    dt = _parse_publication_date(str(pub_date))
    if not dt:
        return True  # Include if parsing fails

    cutoff = datetime.utcnow() - timedelta(days=months * 30)
    return dt >= cutoff


# ---------------------------------------------------------------------------
# Domain relevance filtering
# ---------------------------------------------------------------------------

def is_domain_relevant(
    text: str,
    domain: str,
    topic_terms: dict | None = None,
) -> bool:
    """Check if paper is relevant to domain/topic using AND logic.

    Requires BOTH:
      1. At least one topic-specific term match (target_classes or motif_terms)
      2. At least one domain-level relevance term match
    AND none of the avoid_terms or BAD_TERMS.

    This mirrors the VLAB2 is_biomed_relevant() two-tier gate that prevents
    generic biology papers from passing on a single common keyword.
    """
    if not text:
        return False

    t = text.lower()

    # Hard reject: BAD_TERMS
    if any(term in t for term in BAD_TERMS):
        log.debug("Rejected by BAD_TERMS: %s", text[:80])
        return False

    # Hard reject: topic avoid_terms
    avoid_terms = topic_terms.get("avoid_terms", []) if topic_terms else []
    if any(term.lower() in t for term in avoid_terms):
        log.debug("Rejected by avoid_terms: %s", text[:80])
        return False

    # Topic-specific terms (core identifiers for this topic)
    target_classes = topic_terms.get("target_classes", []) if topic_terms else []
    motif_terms_list = topic_terms.get("motif_terms", []) if topic_terms else []

    # Domain-level terms (broader field relevance)
    domain_data = get_domain_terms(domain)
    domain_relevance = domain_data.get("relevance_terms", [])

    has_topic_match = any(term.lower() in t for term in target_classes + motif_terms_list)
    has_domain_match = any(term.lower() in t for term in domain_relevance)

    # AND gate: require topic match AND domain match (when both are configured)
    if target_classes or motif_terms_list:
        if domain_relevance:
            return has_topic_match and has_domain_match
        return has_topic_match

    # Fallback: only domain terms configured
    if domain_relevance:
        return has_domain_match

    # No terms configured — pass by default
    return True


# ---------------------------------------------------------------------------
# FAISS append (shared with VLAB2)
# ---------------------------------------------------------------------------

def append_to_faiss(docs: List) -> int:
    """Append documents to FAISS index (shared with VLAB2)."""
    if not docs or FAISS is None or CachedSentenceTransformerEmbeddings is None:
        return 0




    try:
        # Initialize Document if not available
        if Document is None:
            class Document:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata








        embeddings = CachedSentenceTransformerEmbeddings()
        index_path = Path(FAISS_INDEX_PATH)
        index_path.parent.mkdir(parents=True, exist_ok=True)



        with _lock:
            try:




                db = FAISS.load_local(
                    str(index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )






                # Simple deduplication
                existing_texts = set()
                try:
                    for doc in db.docstore._dict.values():
                        existing_texts.add(doc.page_content[:200])
                except Exception:
                    pass



                unique_docs = []
                for d in docs:
                    if d.page_content[:200] not in existing_texts:
                        unique_docs.append(d)
                        existing_texts.add(d.page_content[:200])




                if not unique_docs:
                    return 0



                db.add_documents(unique_docs)
                db.save_local(str(index_path))
                return len(unique_docs)




            except Exception as e:
                log.warning("Creating new FAISS index at %s: %s", index_path, e)

                db = FAISS.from_documents(docs, embeddings)
                db.save_local(str(index_path))
                return len(docs)
    except Exception as e:
        log.error("Error in append_to_faiss: %s", str(e))
        return 0

# ---------------------------------------------------------------------------
# Streaming worker
# ---------------------------------------------------------------------------

def fetch_europepmc_papers(query: str, limit: int = 25, time_window_months: int = 240, since_date: str | None = None) -> List[dict]:
    """Fetch papers from Europe PMC using full-text search.

    Europe PMC indexes bioRxiv/medRxiv preprints and supports proper Boolean
    search, unlike the bioRxiv /details/ endpoint which only returns
    chronological listings without search capability.

    Results are sorted by first publication date (newest first). When the
    time window is unlimited (<= 0), the request pages through results so
    older literature is not missed (capped at 500 results).
    If since_date is provided (YYYY-MM-DD), only papers with
    FIRST_PDATE >= since_date are fetched (used for unlimited windows to avoid
    re-fetching already-seen papers).
    """
    import urllib.request
    import urllib.parse
    import json

    results = []
    max_pages = 5
    try:
        # Build date filter
        date_filter = ""
        if time_window_months > 0:
            cutoff = datetime.utcnow() - timedelta(days=time_window_months * 30)
            date_filter = f' AND (FIRST_PDATE:[{cutoff.strftime("%Y-%m-%d")} TO *])'
        elif time_window_months <= 0 and since_date is not None:
            # Since date provided: fetch papers with FIRST_PDATE >= since_date
            date_filter = f' AND (FIRST_PDATE:[{since_date} TO *])'
        # else: no date filter (fetch all)

        # SRC:PPR filters to preprints (bioRxiv, medRxiv, etc.)
        search_query = f"({query}){date_filter} AND SRC:PPR"

        page = 1
        while len(results) < limit and page <= max_pages:
            params = urllib.parse.urlencode({
                "query": search_query,
                "format": "json",
                "pageSize": str(min(limit - len(results), 100)),
                "resultType": "core",
                "sort": "FIRST_PDATE desc",
                "page": str(page),
            })
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}"

            req = urllib.request.Request(url, headers={"User-Agent": "JournalClubPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            items = data.get("resultList", {}).get("result", [])
            if not items:
                break

            for item in items:
                title = item.get("title", "")
                abstract = item.get("abstractText", "")
                doi = item.get("doi", "")
                pub_date = item.get("firstPublicationDate", "")
                year = pub_date.split("-")[0] if pub_date else ""
                authors = []
                for author in (item.get("authorList", {}) or {}).get("author", []) or []:
                    name = author.get("fullName", "")
                    if name:
                        authors.append(name)

                results.append({
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "year": year,
                    "publication_date": pub_date,
                    "source": item.get("source", "europepmc"),
                    "pmid": item.get("pmid", ""),
                    "url": f"https://doi.org/{doi}" if doi else "",
                    "authors": authors,
                    "citation_count": item.get("citedByCount") or 0,
                })

                if len(results) >= limit:
                    break

            page += 1

    except Exception as e:
        log.warning("Europe PMC search failed for query '%s': %s", query, e)

    return results

def ingest_into_analysis(memory, paperwork):
    """Compute summary/gap analysis/quality scores and persist them to memory."""
    def compute_summary_local(paperwork):
        from .paper_analyzer import generate_summary
        return generate_summary(paperwork)

    def compute_gaps_local(paperwork):
        from .paper_analyzer import analyze_gaps
        return analyze_gaps(paperwork)

    def compute_quality_scores(paperwork, gaps):
        from .paper_analyzer import score_paper_quality
        try:
            return score_paper_quality(paperwork, gaps)
        except Exception as e:
            log.warning(f"Error computing quality scores for %s: %s", paperwork.get("title", "Unknown"), e)
            return {}

    try:
        summary = compute_summary_local(paperwork)
        gaps = compute_gaps_local(paperwork)
        quality_scores = compute_quality_scores(paperwork, gaps)

        # update_paper_analysis matches raw doi/pmid/title fields — do NOT
        # pass the normalized _paper_key() here or the lookup silently fails.
        paper_key = (
            paperwork.get("doi")
            or paperwork.get("pmid")
            or paperwork.get("title", "")
        )

        # Update orig memory instance
        memory.update_paper_analysis(
            paper_key,
            summary=summary,
            gap_analysis=gaps,
            quality_scores=quality_scores
        )

        # Force non-blocking checkpoint during analytical pipeline
        memory.save()

    except Exception as e:
        log.warning("Analysis error for paper '%s': %s", paperwork.get("title", "Unknown"), e)




def stream_papers(
    topic_name: str,
    domain: str,
    queries: List[str],
    stop_event: threading.Event,
    interval: int = STREAM_INTERVAL,
    batch_size: int = STREAM_BATCH_SIZE,
    time_window_months: int = DEFAULT_TIME_WINDOW_MONTHS,
    memory: JournalClubMemory | None = None,
):
    """Continuously pull papers for a topic and ingest into memory."""

    if memory is None:
        memory = JournalClubMemory()

    topic_config = get_topic_config(topic_name)
    topic_terms = topic_config.get("domain_terms", {}) if topic_config else {}

    # Deduplicate anything left over from earlier runs before ingesting new papers
    memory.deduplicate_papers()

    log.info("Streaming worker active for topic: %s (domain: %s)", topic_name, domain)

    idle_cycles = 0
    total_cycles = 0

    try:
        while not stop_event.is_set():  # Add accuracy-first salvaging regardless of stop signals
            total_cycles += 1

            try:
                all_new_papers = []
                total_candidates = 0

                # Determine if we should use a bookmark (only when the time
                # window is unlimited). The bookmark records the exclusive
                # lower bound date so we don't re-fetch already-seen preprints.
                use_bookmark = time_window_months <= 0

                for query in queries:
                    since_date = None
                    s2_gated = False
                    if use_bookmark:
                        bookmark_key = f"bookmark:{topic_name}:{query}"
                        since_date = memory.get_bookmark(bookmark_key)
                        # Once the Europe PMC bookmark exists (initial
                        # backfill is done), gate Semantic Scholar: it has
                        # no date cursor and would otherwise re-fetch the
                        # same candidates (and can time out) every cycle.
                        s2_gated = since_date is not None

                    papers = []
                    if ENABLE_SEMANTIC_SCHOLAR and not s2_gated and cached_semantic_search is not None:
                        try:
                            papers = cached_semantic_search(query, limit=batch_size)
                        except TypeError:
                            papers = cached_semantic_search(query, batch_size)

                    # Europe PMC search (proper full-text search of preprints)
                    epmc_papers = fetch_europepmc_papers(
                        query,
                        limit=batch_size,
                        time_window_months=time_window_months,
                        since_date=since_date,
                    )
                    papers = (papers or []) + epmc_papers
                    total_candidates += len(papers)

                    for p in papers:
                        if not isinstance(p, dict):
                            continue

                        pkey = _paper_key(p)

                        if pkey in _stream_cache:
                            continue

                        # Skip papers already stored in memory (survives restarts)
                        if memory.paper_exists(p):
                            continue

                        # Time filtering
                        if not _is_within_time_window(p, time_window_months):
                            continue

                        # Domain relevance filtering
                        combined_text = f"{p.get('title', '')}\n{p.get('abstract', '')}"
                        if not is_domain_relevant(combined_text, domain, topic_terms):
                            log.debug("Filtered non-relevant paper: %s", p.get("title", "")[:80])
                            continue

                        # Ingest into memory
                        record = memory.ingest_paper(
                            p,
                            topic_name=topic_name,
                            domain=domain,
                        )

                        # Analyze paper inline (blocking call)
                        ingest_into_analysis(memory, p)

                        if record:
                            _stream_cache.add(pkey)
                            all_new_papers.append(p)

                    # Advance the Europe PMC bookmark to just past the newest
                    # fetched date (exclusive lower bound), so the next cycle
                    # requests only strictly newer preprints.
                    if use_bookmark and epmc_papers:
                        max_date = ""
                        for p in epmc_papers:
                            d = p.get("publication_date") or ""
                            if d > max_date:
                                max_date = d
                        if max_date:
                            parsed = _parse_publication_date(max_date)
                            if parsed:
                                next_since = (parsed + timedelta(days=1)).strftime("%Y-%m-%d")
                                current = memory.get_bookmark(bookmark_key)
                                if current is None or next_since > current:
                                    memory.set_bookmark(bookmark_key, next_since)

                if all_new_papers:
                    # Also append to FAISS if available
                    if Document is not None:
                        docs = [
                            Document(
                                page_content=f"{p.get('title', '')}\n\n{p.get('abstract', '')}",
                                metadata={
                                    "title": p.get("title"),
                                    "source": "journal_club",
                                    "topic": topic_name,
                                    "domain": domain,
                                }
                            )
                            for p in all_new_papers
                        ]
                        added = append_to_faiss(docs)
                        if added > 0:
                            log.debug("Added %d records to FAISS index", added)

                    memory.save()
                    idle_cycles = 0
                    log.info(
                        "Streaming cycle %d: fetched %d candidates, added %d new papers for topic: %s",
                        total_cycles,
                        total_candidates,
                        len(all_new_papers),
                        topic_name,
                    )
                else:
                    idle_cycles += 1
                    log.info(
                        "Streaming cycle %d: fetched %d candidates, no new papers for topic: %s (idle cycle %d)",
                        total_cycles,
                        total_candidates,
                        topic_name,
                        idle_cycles,
                    )

                if STREAM_MAX_IDLE > 0 and idle_cycles >= STREAM_MAX_IDLE:
                    log.info(
                        "Stopping stream for topic after %d idle cycles: %s",
                        idle_cycles,
                        topic_name,
                    )
                    break

                if STREAM_MAX_CYCLES > 0 and total_cycles >= STREAM_MAX_CYCLES:
                    log.info(
                        "Stopping stream for topic after %d total cycles: %s",
                        total_cycles,
                        topic_name,
                    )
                    break

            except Exception as e:
                log.warning("Streaming ingestion error for topic '%s': %s", topic_name, e)
                # Ensure we save any progress made even if an error occurs
                memory.save()


            stop_event.wait(interval)

    except KeyboardInterrupt:
        log.info(
            "KeyboardInterrupt caught; ensuring memory saved before stopping stream. Current papers: %d",
            memory.get_statistics().get('total_papers', 0),
        )
        memory.save()






    finally:
        with _lock:
            key = f"{topic_name}::{domain}"
            _active_queries.pop(key, None)
            _active_threads.pop(key, None)

        # Single memory save at the end
        memory.save()
        log.info("Streaming worker stopped for topic: %s", topic_name)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _stream_key(topic_name: str, domain: str) -> str:
    return f"{topic_name}::{domain}".lower()


def start_streaming(
    topic_name: str,
    domain: str,
    queries: List[str],
    time_window_months: int = DEFAULT_TIME_WINDOW_MONTHS,
    memory: JournalClubMemory | None = None,
) -> bool:
    """Start streaming literature ingestion for a topic."""

    if not queries:
        log.warning("Refusing to start streaming for empty queries.")
        return False

    key = _stream_key(topic_name, domain)

    with _lock:
        if key in _active_queries:
            log.info("Streaming already active for topic/domain: %s", topic_name)
            return False

        stop_event = threading.Event()
        _active_queries[key] = stop_event

        thread = threading.Thread(
            target=stream_papers,
            args=(topic_name, domain, queries, stop_event),
            kwargs={
                "time_window_months": time_window_months,
                "memory": memory,
            },
            daemon=True,
            name=f"journal-club-stream::{key[:40]}",
        )
        thread.setName(f"stream_{topic_name[:15]}_{memory.path[-10:]}")

        _active_threads[key] = thread
        thread.start()

    log.info("Started streaming literature ingestion for topic: %s", topic_name)
    return True


def stop_streaming(topic_name: str, domain: str) -> bool:
    """Stop streaming for a topic."""
    key = _stream_key(topic_name, domain)

    with _lock:
        stop_event = _active_queries.get(key)

        if stop_event is None:
            return False

        stop_event.set()
        return True


def stop_all_streaming() -> int:
    """Stop all streaming."""
    with _lock:
        events = list(_active_queries.values())

        for ev in events:
            ev.set()

    # Clean up LLM clients to prevent segfaults
    try:
        cleanup_llm_clients()
    except Exception as e:
        log.warning(f"Error during LLM cleanup: {e}")

    return len(events)


def active_streams() -> list[str]:
    """Get list of active streams."""
    with _lock:
        return list(_active_queries.keys())


def start_all_topics(memory: JournalClubMemory | None = None) -> int:
    """Start streaming for all configured topics."""
    topics = load_topic_config()
    started = 0

    for topic in topics:
        topic_name = topic.get("name")
        domain = topic.get("domain", "general")
        queries = topic.get("seed_queries", [])
        time_window = topic.get("time_window_months", DEFAULT_TIME_WINDOW_MONTHS)

        if topic_name and queries:
            if start_streaming(topic_name, domain, queries, time_window, memory):
                started += 1

    log.info("Started streaming for %d topics", started)
    return started
