"""
recommendation_engine.py

Recommendation engine for Journal Club.

Responsibilities:
  - Detect foundational papers (highly-cited, older papers)
  - Detect conflicting papers (contradictory conclusions)
  - Recommend related reading based on semantic similarity
  - Cluster papers by theme
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Try to import from VLAB2
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
    CachedSentenceTransformerEmbeddings = None
    cached_semantic_search = None
    FAISS = None
    Document = None

log = logging.getLogger("journal_club.recommendations")


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

# Foundational paper criteria
FOUNDATIONAL_MIN_CITATIONS = 50
FOUNDATIONAL_MIN_YEARS_OLD = 3
FOUNDATIONAL_CITATION_THRESHOLD = 100

# Fine-tuned model configuration
USE_FINETUNED = os.getenv("JOURNAL_CLUB_USE_FINETUNED", "0") == "1"
FINETUNED_MODEL_PATH = os.getenv(
    "JOURNAL_CLUB_FINETUNED_MODEL_PATH",
    "training/journal_club_merged_model"
)
FALLBACK_TO_BASE = os.getenv("JOURNAL_CLUB_FALLBACK_TO_BASE", "1") == "1"


from .paper_analyzer import get_llm_client, _get_response_text


# ---------------------------------------------------------------------------
# Foundational Paper Detection
# ---------------------------------------------------------------------------

def is_foundational_paper(paper: Dict[str, Any]) -> bool:
    """Determine if a paper is foundational (highly-cited, older)."""

    citation_count = paper.get("citation_count") or 0
    year = paper.get("year")

    if not year:
        return False

    try:
        year = int(year)
        years_old = datetime.utcnow().year - year

        # Must be old enough
        if years_old < FOUNDATIONAL_MIN_YEARS_OLD:
            return False

        # Must have sufficient citations
        if citation_count >= FOUNDATIONAL_CITATION_THRESHOLD:
            return True

        if citation_count >= FOUNDATIONAL_MIN_CITATIONS and years_old >= 5:
            return True

    except (ValueError, TypeError):
        return False

    return False


def find_foundational_papers(
    papers: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Find foundational papers from a list."""

    foundational = [p for p in papers if is_foundational_paper(p)]

    # Sort by citation count (descending)
    foundational.sort(
        key=lambda p: p.get("citation_count", 0),
        reverse=True
    )

    return foundational[:limit]


# ---------------------------------------------------------------------------
# Conflicting Paper Detection
# ---------------------------------------------------------------------------

def detect_conflicting_papers(
    paper: Dict[str, Any],
    other_papers: List[Dict[str, Any]],
    llm_client=None,
) -> List[Dict[str, Any]]:
    """Detect papers with contradictory conclusions."""

    if not other_papers:
        return []

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Rule-based detection first
    conflicting = _rule_based_conflict_detection(paper, other_papers)

    # If LLM available, do deeper analysis
    if llm_client is None:
        llm_client = get_llm_client()

    if llm_client and conflicting:
        conflicting = _llm_conflict_verification(
            paper,
            conflicting,
            llm_client
        )

    return conflicting


def _rule_based_conflict_detection(
    paper: Dict[str, Any],
    other_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rule-based conflict detection."""

    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()

    # Look for contradiction indicators
    contradiction_terms = [
        "contradicts",
        "contrary to",
        "inconsistent with",
        "disagrees with",
        "fails to replicate",
        "no effect",
        "no significant",
        "not significant",
    ]

    # Look for agreement indicators
    agreement_terms = [
        "confirms",
        "replicates",
        "consistent with",
        "agrees with",
        "supports",
        "validates",
    ]

    conflicting = []

    for other in other_papers:
        other_title = other.get("title", "").lower()
        other_abstract = other.get("abstract", "").lower()
        other_combined = f"{other_title} {other_abstract}"

        # Skip self
        if other.get("doi") == paper.get("doi"):
            continue

        # Check for contradiction indicators
        has_contradiction = any(term in other_combined for term in contradiction_terms)

        # Check if it references similar concepts
        title_words = set(title.split())
        other_title_words = set(other_title.split())
        overlap = len(title_words & other_title_words)

        if has_contradiction and overlap >= 2:
            conflicting.append(other)

    return conflicting


def _llm_conflict_verification(
    paper: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    llm_client,
) -> List[Dict[str, Any]]:
    """Use LLM to verify conflicts."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    verified = []

    for candidate in candidates:
        candidate_title = candidate.get("title", "")
        candidate_abstract = candidate.get("abstract", "")

        prompt = f"""Compare these two papers and determine if they have contradictory conclusions:

Paper A:
Title: {title}
Abstract: {abstract}

Paper B:
Title: {candidate_title}
Abstract: {candidate_abstract}

Do these papers have contradictory or conflicting conclusions? Answer "yes" or "no" with a brief explanation.
"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="You are an expert at identifying scientific contradictions."),
                HumanMessage(content=prompt),
            ]

            response = llm_client.invoke(messages)
            resp_str = _get_response_text(response).lower()

            if "yes" in resp_str:
                verified.append(candidate)

        except Exception as e:
            log.warning("LLM conflict verification failed: %s", e)
            # Include candidate if LLM fails
            verified.append(candidate)

    return verified


# ---------------------------------------------------------------------------
# Related Reading
# ---------------------------------------------------------------------------

def recommend_related_reading(
    paper: Dict[str, Any],
    faiss_index_path: str = FAISS_INDEX_PATH,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Recommend related papers using semantic search."""

    if cached_semantic_search is None:
        log.warning("Semantic search not available for recommendations")
        return []

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Build query from title and key terms
    query = f"{title} {abstract[:200]}"

    try:
        related = cached_semantic_search(query, limit=top_k * 2)  # Get more, filter later

        # Filter out self
        doi = paper.get("doi", "").lower()
        related = [p for p in related if p.get("doi", "").lower() != doi]

        return related[:top_k]

    except Exception as e:
        log.warning("Related reading search failed: %s", e)
        return []


def recommend_related_from_faiss(
    paper: Dict[str, Any],
    faiss_index_path: str = FAISS_INDEX_PATH,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Recommend related papers using FAISS similarity search."""

    if FAISS is None or CachedSentenceTransformerEmbeddings is None:
        return []

    try:
        embeddings = CachedSentenceTransformerEmbeddings()
        index_path = Path(faiss_index_path)

        if not index_path.exists():
            log.warning("FAISS index not found at %s", faiss_index_path)
            return []

        db = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Create query document
        query_text = f"{paper.get('title', '')}\n{paper.get('abstract', '')}"
        query_doc = Document(page_content=query_text)

        # Search
        results = db.similarity_search_with_score(query_text, k=top_k + 10)

        # Convert to paper-like dicts, filter self
        related = []
        seen_dois = {paper.get("doi", "").lower()}

        for doc, score in results:
            metadata = doc.metadata or {}
            doi = metadata.get("doi", "").lower()

            if doi and doi in seen_dois:
                continue

            if doi:
                seen_dois.add(doi)

            related.append({
                "title": metadata.get("title", ""),
                "abstract": doc.page_content,
                "score": float(score),
                "source": metadata.get("source", "faiss"),
                "doi": metadata.get("doi"),
            })

        return related[:top_k]

    except Exception as e:
        log.warning("FAISS recommendation failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Theme Clustering
# ---------------------------------------------------------------------------

def cluster_papers_by_theme(
    papers: List[Dict[str, Any]],
    n_clusters: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Cluster papers by theme using simple keyword-based clustering."""

    if not papers:
        return {}

    # Extract keywords from titles and abstracts
    paper_keywords = []
    for p in papers:
        title = p.get("title", "").lower()
        abstract = p.get("abstract", "").lower()
        combined = f"{title} {abstract}"

        # Extract meaningful words (length > 3)
        words = [w for w in combined.split() if len(w) > 4]
        paper_keywords.append(words)

    # Build keyword frequency across all papers
    all_keywords = defaultdict(int)
    for keywords in paper_keywords:
        for kw in keywords:
            all_keywords[kw] += 1

    # Get top keywords as themes
    top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
    theme_keywords = [kw for kw, count in top_keywords[:n_clusters]]

    # Assign papers to themes
    clusters = {f"theme_{i}": [] for i in range(n_clusters)}

    for i, (paper, keywords) in enumerate(zip(papers, paper_keywords)):
        # Find best matching theme
        best_theme = 0
        best_match = 0

        for j, theme_kw in enumerate(theme_keywords):
            if theme_kw in keywords:
                best_theme = j
                best_match += 1

        theme_name = f"theme_{best_theme}"
        clusters[theme_name].append(paper)

    # Rename themes with actual keywords
    renamed_clusters = {}
    for i, (theme_name, theme_papers) in enumerate(clusters.items()):
        if i < len(theme_keywords):
            new_name = theme_keywords[i].replace("_", " ").title()
        else:
            new_name = f"Other"

        renamed_clusters[new_name] = theme_papers

    return renamed_clusters


# ---------------------------------------------------------------------------
# Full Recommendation Pipeline
# ---------------------------------------------------------------------------

def generate_recommendations(
    paper: Dict[str, Any],
    all_papers: List[Dict[str, Any]],
    faiss_index_path: str = FAISS_INDEX_PATH,
    llm_client=None,
) -> Dict[str, Any]:
    """Generate all recommendations for a paper."""

    log.info("Generating recommendations for: %s", paper.get("title", "")[:60])

    # Foundational papers in the same topic
    foundational = find_foundational_papers(all_papers, limit=5)

    # Conflicting papers
    other_papers = [p for p in all_papers if p.get("doi") != paper.get("doi")]
    conflicting = detect_conflicting_papers(paper, other_papers, llm_client)

    # Related reading
    related_faiss = recommend_related_from_faiss(paper, faiss_index_path, top_k=5)
    related_semantic = recommend_related_reading(paper, faiss_index_path, top_k=5)

    # Combine and deduplicate related papers
    related = []
    seen_dois = set()

    for p in related_faiss + related_semantic:
        doi = p.get("doi", "")
        if doi and doi.lower() in seen_dois:
            continue
        if doi:
            seen_dois.add(doi.lower())
        related.append(p)

    return {
        "foundational": foundational,
        "conflicting": conflicting,
        "related": related[:10],
    }


def batch_recommendations(
    papers: List[Dict[str, Any]],
    faiss_index_path: str = FAISS_INDEX_PATH,
    llm_client=None,
) -> Dict[str, Dict[str, Any]]:
    """Generate recommendations for a batch of papers."""

    recommendations = {}

    for i, paper in enumerate(papers):
        log.info("Generating recommendations for paper %d/%d", i + 1, len(papers))

        try:
            recs = generate_recommendations(
                paper,
                papers,
                faiss_index_path,
                llm_client,
            )

            key = paper.get("doi") or paper.get("title") or f"paper_{i}"
            recommendations[key] = recs

        except Exception as e:
            log.warning("Failed to generate recommendations for paper %d: %s", i, e)

    return recommendations
