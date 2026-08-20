"""
streaming_agent.py

Background literature ingestion for Journal Club.

Responsibilities:
  - Start one background stream per topic/domain profile
  - Fetch papers using semantic search (shared with VLAB2)
  - Filter for domain-specific relevance using configurable terms
  - Filter by publication date (configurable time window)
  - Deduplicate papers by DOI, normalized title
  - Append genuinely new documents to FAISS (shared with VLAB2)
  - Ingest papers into JournalClubMemory

Important behaviour:
  - "No new papers" is logged at DEBUG, not INFO
  - FAISS path is configurable (default: VLAB2 location)
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

# Try to import from VLAB2, fallback to local implementation
try:
    import sys
    vlab2_path = Path(__file__).parents[2] / "VLAB2"
    if vlab2_path.exists():
        # Add the PARENT of VLAB2 so that 'VLAB2' is importable as a namespace package

        sys.path.insert(0, str(vlab2_path.parent))

    from VLAB2.research.research_agent_adaptive import (
        CachedSentenceTransformerEmbeddings,
        cached_semantic_search,
    )
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError:
    # Fallback implementations if VLAB2 not available
    CachedSentenceTransformerEmbeddings = None
    cached_semantic_search = None
    FAISS = None
    Document = None

from .literature_memory import JournalClubMemory

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

DEFAULT_FAISS_INDEX_PATH = str(
    Path(__file__).parents[1] / "cache" / "faiss_index"
)

FAISS_INDEX_PATH = os.getenv(
    "JOURNAL_CLUB_FAISS_INDEX_PATH",
    DEFAULT_FAISS_INDEX_PATH
)

STREAM_INTERVAL = int(os.getenv("JOURNAL_CLUB_STREAM_INTERVAL", "30"))
STREAM_BATCH_SIZE = int(os.getenv("JOURNAL_CLUB_STREAM_BATCH_SIZE", "20"))
STREAM_MAX_IDLE = int(os.getenv("JOURNAL_CLUB_STREAM_MAX_IDLE", "0"))
STREAM_MAX_CYCLES = int(os.getenv("JOURNAL_CLUB_STREAM_MAX_CYCLES", "0"))
DEDUP_ABSTRACT_PREFIX_LEN = int(os.getenv("JOURNAL_CLUB_DEDUP_ABSTRACT_PREFIX_LEN", "500"))

DEFAULT_TIME_WINDOW_MONTHS = int(os.getenv("JOURNAL_CLUB_TIME_WINDOW_MONTHS", "12"))

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
    """Check if paper is within time window."""
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
    """Check if paper is relevant to domain/topic."""
    if not text:
        return False

    t = text.lower()

    # Get domain terms
    domain_terms = get_domain_terms(domain)
    relevance_terms = list(domain_terms.get("relevance_terms", []))

    # Add topic-specific terms if provided
    if topic_terms:
        relevance_terms.extend(topic_terms.get("target_classes", []))
        relevance_terms.extend(topic_terms.get("motif_terms", []))

    # Check for avoid terms first
    avoid_terms = topic_terms.get("avoid_terms", []) if topic_terms else []
    if any(term in t for term in avoid_terms):
        return False

    # If no specific relevance terms are defined, pass by default
    if not relevance_terms:
        return True

    # Check for relevance terms
    return any(term.lower() in t for term in relevance_terms)


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

                # Create empty Document templates if needed for the vector store
                dummy_docs = [Document("dummy_title", {})]

                db = FAISS.from_documents(docs, embeddings)
                db.save_local(str(index_path))
                return len(docs)
    except Exception as e:
        log.error("Error in append_to_faiss: %s", str(e))
        return 0

# ---------------------------------------------------------------------------
# Streaming worker
# ---------------------------------------------------------------------------

def fetch_biorxiv_preprints(query: str, limit: int = 10) -> List[dict]:
    """Fetch recent preprints from BioRxiv/MedRxiv API."""
    import urllib.request
    import json

    results = []
    try:
        url = "https://api.biorxiv.org/details/biorxiv/2024-01-01/2026-12-31/0/json"
        req = urllib.request.Request(url, headers={"User-Agent": "JournalClubPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            messages = data.get("collection", [])
            keywords = [k.lower() for k in query.split() if len(k) > 3]
            if not keywords:
                keywords = [query.lower()]

            for item in messages:
                title = item.get("title", "").lower()
                abstract = item.get("abstract", "").lower()
                text = f"{title} {abstract}"
                if any(kw in text for kw in keywords):
                    results.append({
                        "title": item.get("title"),
                        "abstract": item.get("abstract"),
                        "doi": item.get("doi"),
                        "year": item.get("date", "").split("-")[0] if item.get("date") else "2025",
                        "publication_date": item.get("date"),
                        "source": "biorxiv",
                    })
                    if len(results) >= limit:
                        break
    except Exception as e:
        log.debug("BioRxiv preprint fetch skipped/failed: %s", e)

    return results

def ingest_into_analysis(local_memory, paperwork, key, stop_event, topic_name, time_window_months, memory):
    def compute_summary_local(paperwork):
        from core.paper_analyzer import generate_summary
        return generate_summary(paperwork)





    def compute_gaps_local(paperwork):
        from core.paper_analyzer import analyze_gaps
        return analyze_gaps(paperwork)







    def compute_quality_scores(paperwork, gaps):
        from core.paper_analyzer import score_paper_quality
        try:
            return score_paper_quality(paperwork, gaps)

        except Exception as e:
            log.warning(f"Error computing quality scores for {key}: {e}")
            return {}




    try:
        summary = compute_summary_local(paperwork)

        domain = paperwork.get("domain", "general")
        gaps = compute_gaps_local(paperwork)
        quality_scores = compute_quality_scores(paperwork, gaps)

        # Add analysis results to the record






        record = {
            "summary": summary,
            "critique": "Not computed yet",
            "gap_analysis": gaps,
            "quality_scores": quality_scores
        }






        # Quick update to the literature memory
        local_memory.memory["updated_cache"].setdefault(topic_name, []).append({
            "key": key,
            "updates": record
        })










        # Update orig memory instance
        memory.update_paper_analysis(
            key,
            summary=summary,
            gap_analysis=gaps,
            quality_scores=quality_scores
        )





        # Force non-blocking checkpoint during analytical pipeline
        local_memory.save()



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

    log.info("Streaming worker active for topic: %s (domain: %s)", topic_name, domain)

    idle_cycles = 0
    total_cycles = 0

    try:


        # Use a local memory object and periodically refresh it from the file
        # to ensure consistency even across different thread executions.
        local_memory = JournalClubMemory()

        while not stop_event.is_set():  # Add accuracy-first salvaging regardless of stop signals
            total_cycles += 1

            try:
                all_new_papers = []

                for query in queries:
                    papers = []
                    if cached_semantic_search is not None:
                        try:
                            papers = cached_semantic_search(query, limit=batch_size)
                        except TypeError:
                            papers = cached_semantic_search(query, batch_size)

                    # BioRxiv fallback/supplement
                    preprint_papers = fetch_biorxiv_preprints(query, limit=5)
                    papers = (papers or []) + preprint_papers

                    for p in papers:
                        if not isinstance(p, dict):
                            continue

                        pkey = _paper_key(p)

                        if pkey in _stream_cache:
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
                            time_window_months=time_window_months,
                        )

                        # Asynchronously in queue
                        ingest_into_analysis(
                            local_memory,
                            p,
                            pkey,
                            stop_event,
                            topic_name,

                            time_window_months,
                            memory  # Pass memory reference explicitly
                        )

                        if record:
                            _stream_cache.add(pkey)
                            all_new_papers.append(p)

                if all_new_papers:
                    # Also append to FAISS if available

                    if Document is not None or hasattr(Document, '__class__'):
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

                    local_memory.save()
                    idle_cycles = 0
                    log.info(
                        "Streaming: added %d new papers for topic: %s",
                        len(all_new_papers),
                        topic_name,
                    )
                else:
                    idle_cycles += 1
                    log.debug(
                        "Streaming: no new papers for topic: %s (idle cycle %d)",
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
                local_memory.save()


            stop_event.wait(interval)

    except KeyboardInterrupt:
        log.info(f"KeyboardInterrupt caught; ensuring memory saved before stopping stream. Current record: {len(local_memory.get_statistics().get('papers', []))}")
        local_memory.save()






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
