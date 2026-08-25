from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List

import requests
from Bio import Entrez
from dotenv import load_dotenv

log = logging.getLogger("journal_club.researcher")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL", "journal.club@example.com")

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("S2_API_KEY")

_session = requests.Session()

headers = {
    "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
    "User-Agent": "JournalClub/1.0",
}

_semantic_cache = {}


def cached_semantic_search(query, limit):
    cache_key = (str(query or "").strip().lower(), int(limit or 25))

    if cache_key in _semantic_cache:
        return _semantic_cache[cache_key]

    result = search_semantic_scholar(query, limit=limit)

    if result:
        _semantic_cache[cache_key] = result

    return result


_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
    return _model


from langchain_core.embeddings import Embeddings


class CachedSentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = get_embedding_model()

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


_last_call_time = 0
_lock = threading.Lock()
MIN_INTERVAL = 1.0


def rate_limited_get(url, **kwargs):
    global _last_call_time
    with _lock:
        now = time.time()
        elapsed = now - _last_call_time

        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        resp = _session.get(url, **kwargs)
        _last_call_time = time.time()

    return resp


def _faiss_index_path() -> str:
    try:
        from .streaming_literature_agent import get_faiss_index_path
        return get_faiss_index_path()
    except Exception:
        return (
            os.getenv("JOURNAL_CLUB_FAISS_INDEX_PATH")
            or os.getenv("FAISS_INDEX_PATH")
            or os.path.join(os.path.dirname(__file__), "..", "cache", "faiss_index")
        )


def search_local_db(query: str) -> List:
    try:
        from langchain_community.vectorstores import FAISS

        index_path = _faiss_index_path()

        if not os.path.exists(index_path):
            return []

        embeddings = CachedSentenceTransformerEmbeddings()
        db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return db.similarity_search(query, k=10)

    except Exception as e:
        log.warning("Local DB error: %s", e)
        return []


def search_semantic_scholar(query: str, limit: int = 25) -> List:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,year,url,externalIds",
    }

    for attempt in range(5):
        try:
            resp = rate_limited_get(
                SEMANTIC_SCHOLAR_API_URL,
                params=params,
                headers=headers,
                timeout=10,
            )

            if resp.status_code == 429:
                time.sleep(min(2**attempt, 30))
                continue

            if resp.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                continue

            resp.raise_for_status()
            data = resp.json().get("data", []) or []

            out = []
            for p in data:
                ext = p.get("externalIds") or {}
                row = {
                    "title": p.get("title"),
                    "abstract": p.get("abstract"),
                    "year": p.get("year"),
                    "url": p.get("url"),
                    "doi": ext.get("DOI"),
                    "pmid": ext.get("PubMed"),
                    "source": "semantic_scholar",
                }
                out.append(row)

            return out

        except Exception as e:
            log.warning("Semantic Scholar error: %s", e)
            time.sleep(1)

    return []


def expand_knowledge(topic: str, build_db: bool = False) -> List:
    papers = cached_semantic_search(topic, limit=25)

    if build_db and papers:
        try:
            from langchain_core.documents import Document
            from .streaming_literature_agent import append_to_faiss

            docs = []

            for p in papers:
                title = p.get("title") or ""
                abstract = p.get("abstract") or ""
                text = f"{title}\n\n{abstract}".strip()

                if not text:
                    continue

                doi = p.get("doi") or p.get("DOI")
                year = p.get("year")

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "doi": doi,
                            "url": p.get("url"),
                            "source": p.get("source", "semantic_scholar"),
                            "query": topic,
                            "search_query": topic,
                        },
                    )
                )

            if docs:
                added = append_to_faiss(docs)
                log.info("FAISS appended with %d docs for topic '%s'", added, topic)

        except Exception as e:
            log.warning("FAISS append failed: %s", e)

    return papers