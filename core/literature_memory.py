"""
literature_memory.py

Memory backend for Journal Club (SQLite).

Responsibilities:
  - Store and manage academic papers
  - Provide methods to query, filter, and aggregate papers
  - Persist state to a SQLite database (WAL mode for concurrent readers)
  - Migrate legacy JSON memory files automatically on first use

SQLite replaces the previous single-JSON-file design: concurrent analysis
threads, stream workers, and the web app can now read/write without losing
updates, and queries (filtering, statistics, deduplication) are indexed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from . import config

log = logging.getLogger("journal_club.memory")

SCHEMA_VERSION = "journal_club_memory.v2"
PAPER_SCHEMA_VERSION = "journal_club_paper.v1"

_LEGACY_JSON_SUFFIX = ".json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_doi(doi: Any) -> str:
    return str(doi or "").strip().lower().replace(" ", "")


def _norm_title(title: Any) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip().lower())


class JournalClubMemory:
    """SQLite-backed memory for Journal Club papers.

    Public API is compatible with the legacy JSON-backed implementation.
    """

    def __init__(self, path: str | None = None):
        self.save_logger = logging.getLogger("journal_club.memory.save")
        self.stat_logger = logging.getLogger("journal_club.memory.stats")

        self.path = self._resolve_db_path(path)

        # Serializes DB access within this instance. SQLite connections are
        # not thread-safe; analysis threads share one memory instance.
        self._write_lock = threading.RLock()

        self._conn = self._connect(self.path)
        self._init_schema()
        self._migrate_legacy_json(path)
        log.info("JournalClubMemory ready at %s", self.path)

    # ------------------------------------------------------------------
    # Setup / schema
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_db_path(path: str | None) -> str:
        if path is None:
            path = str(config.DEFAULT_MEMORY_PATH)
        p = str(path)
        if p.endswith(_LEGACY_JSON_SUFFIX):
            # A legacy JSON path was supplied (or is set via the env var):
            # use a sibling .db file and migrate the JSON automatically.
            return p[: -len(_LEGACY_JSON_SUFFIX)] + ".db"
        return p

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        dir_name = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    title_norm TEXT NOT NULL,
                    abstract TEXT NOT NULL DEFAULT '',
                    year TEXT NOT NULL DEFAULT '',
                    publication_date TEXT NOT NULL DEFAULT '',
                    doi TEXT NOT NULL DEFAULT '',
                    doi_norm TEXT NOT NULL DEFAULT '',
                    pmid TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    authors TEXT NOT NULL DEFAULT '[]',
                    citation_count INTEGER NOT NULL DEFAULT 0,
                    topic_name TEXT NOT NULL DEFAULT '',
                    domain TEXT NOT NULL DEFAULT '',
                    schema_version TEXT NOT NULL DEFAULT 'journal_club_paper.v1',
                    timestamp TEXT NOT NULL DEFAULT '',
                    summary TEXT,
                    critique TEXT,
                    gap_analysis TEXT,
                    quality_scores TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi_norm);
                CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title_norm);
                CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic_name);
                CREATE INDEX IF NOT EXISTS idx_papers_domain ON papers(domain);

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            self._conn.commit()

    _INSERT_SQL = """
        INSERT INTO papers (
            title, title_norm, abstract, year, publication_date,
            doi, doi_norm, pmid, url, source, authors,
            citation_count, topic_name, domain, schema_version,
            timestamp, summary, critique, gap_analysis, quality_scores
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    @staticmethod
    def _paper_params(
        paper: dict,
        topic_name: str | None = None,
        domain: str | None = None,
        timestamp: str | None = None,
    ) -> tuple:
        """Build the parameter tuple for a paper INSERT (None-safe)."""
        title = paper.get("title") or "untitled"
        return (
            title,
            _norm_title(title),
            paper.get("abstract") or "",
            paper.get("year") or "",
            paper.get("publication_date") or "",
            paper.get("doi") or "",
            _norm_doi(paper.get("doi")),
            paper.get("pmid") or "",
            paper.get("url") or "",
            paper.get("source") or "",
            json.dumps(paper.get("authors") or []),
            paper.get("citation_count") or 0,
            topic_name or "",
            domain or "",
            paper.get("schema_version") or PAPER_SCHEMA_VERSION,
            timestamp or _now_iso(),
            paper.get("summary"),
            paper.get("critique"),
            json.dumps(paper["gap_analysis"]) if paper.get("gap_analysis") is not None else None,
            json.dumps(paper["quality_scores"]) if paper.get("quality_scores") is not None else None,
        )

    def _migrate_legacy_json(self, original_path: str | None) -> None:
        """Migrate a legacy JSON memory file into SQLite (once, atomically)."""
        json_path: Path | None = None
        if original_path and str(original_path).endswith(_LEGACY_JSON_SUFFIX):
            json_path = Path(original_path)
        else:
            default_json = str(self.path)[: -len(".db")] + _LEGACY_JSON_SUFFIX
            if os.path.exists(default_json):
                json_path = Path(default_json)

        if json_path is None or not json_path.exists():
            return

        if self.get_metadata("migrated_json") == str(json_path):
            return

        log.info("Migrating legacy JSON memory %s -> %s", json_path, self.path)
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(
                "Legacy JSON %s could not be read (%s); skipping migration",
                json_path,
                e,
            )
            return
        if not isinstance(data, dict):
            log.warning(
                "Legacy JSON %s has unexpected structure; skipping migration",
                json_path,
            )
            return

        papers = [p for p in data.get("papers", []) if isinstance(p, dict) and p.get("title")]
        rows = [
            self._paper_params(p, topic_name=p.get("topic_name"), domain=p.get("domain"), timestamp=p.get("timestamp"))
            for p in papers
        ]

        # Single transaction: either the whole migration succeeds or nothing does.
        with self._write_lock:
            try:
                self._conn.executemany(self._INSERT_SQL, rows)
                for key in ("training_version", "last_training_date"):
                    if data.get(key) is not None:
                        self._conn.execute(
                            "INSERT INTO metadata(key, value) VALUES(?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (key, str(data[key])),
                        )
                self._conn.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        backup = json_path.with_suffix(_LEGACY_JSON_SUFFIX + ".migrated")
        os.replace(str(json_path), str(backup))
        self.set_metadata("migrated_json", str(backup))
        log.warning(
            "Migrated %d papers; legacy JSON renamed to %s — SQLite is now the source of truth",
            len(rows),
            backup,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> None:
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self._conn.commit()

    def get_metadata(self, key: str, default: Any = None):
        with self._write_lock:
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def get_bookmark(self, key: str) -> str | None:
        """Get a string bookmark (e.g. a Europe PMC fetch cursor) from metadata."""
        value = self.get_metadata(key)
        return value if value not in (None, "") else None

    def set_bookmark(self, key: str, value: str) -> None:
        """Store a string bookmark in metadata."""
        self.set_metadata(key, value)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Commit pending changes (kept for API compatibility with the
        JSON-backed implementation)."""
        with self._write_lock:
            self._compact()
            self._conn.commit()
            self.stat_logger.info(
                "Successfully saved memory: %s, papers: %d",
                self.path,
                self._count(),
            )

    def update_statistics(self, memory: dict | None = None) -> None:
        """Statistics are computed live from SQLite in get_statistics();
        kept for API compatibility."""

    def _compact(self) -> None:
        """Remove invalid paper records (missing required fields)."""
        with self._write_lock:
            cur = self._conn.execute(
                "DELETE FROM papers WHERE title = '' OR schema_version = '' "
                "OR timestamp = '' OR topic_name = ''"
            )
            if cur.rowcount:
                log.info("Pruned %d invalid paper records", cur.rowcount)

    def _count(self) -> int:
        with self._write_lock:
            return self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def close(self) -> None:
        with self._write_lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @property
    def memory(self) -> dict:
        """Legacy dict view of the memory (rebuilt as a snapshot on each
        access). Prefer the query methods; this exists for compatibility."""
        return {
            "papers": self.get_all_papers(),
            "schema_version": SCHEMA_VERSION,
            "statistics": self.get_statistics(),
            "updated_cache": {},
        }

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> dict:
        paper = dict(row)
        for field in ("authors", "gap_analysis", "quality_scores"):
            raw = paper.get(field)
            if isinstance(raw, str) and raw:
                try:
                    paper[field] = json.loads(raw)
                except Exception:
                    pass
        return paper

    # ------------------------------------------------------------------
    # Ingestion / updates
    # ------------------------------------------------------------------

    def ingest_paper(
        self,
        paper: dict | Any,
        topic_name: str | None = None,
        domain: str | None = None,
        time_window_months: int = 240,
    ) -> dict | None:
        """Ingest a paper into memory with its metadata."""
        try:
            title = paper.get("title") or "untitled"
            research_metadata: Dict[str, Any] = {
                "schema_version": paper.get("schema_version") or PAPER_SCHEMA_VERSION,
                "timestamp": _now_iso(),
                "title": title,
                "abstract": paper.get("abstract") or "",
                "year": paper.get("year") or "",
                "publication_date": paper.get("publication_date") or "",
                "doi": paper.get("doi") or "",
                "pmid": paper.get("pmid") or "",
                "url": paper.get("url") or "",
                "source": paper.get("source") or "",
                "authors": paper.get("authors") or [],
                "citation_count": paper.get("citation_count") or 0,
                "topic_name": topic_name,
                "domain": domain,
            }

            # Inject the actual analysis result as soon as it's obtained
            if "summary" in paper:
                research_metadata["summary"] = paper["summary"]
            if "gap_analysis" in paper:
                research_metadata["gap_analysis"] = paper["gap_analysis"]
            if "quality_scores" in paper:
                research_metadata["quality_scores"] = paper["quality_scores"]

            with self._write_lock:
                self._conn.execute(
                    self._INSERT_SQL,
                    self._paper_params(
                        research_metadata,
                        topic_name=topic_name,
                        domain=domain,
                        timestamp=research_metadata["timestamp"],
                    ),
                )
                self._conn.commit()

            self.save_logger.info("Successfully appended paper %s", title[:60])
            return research_metadata
        except Exception as e:
            self.save_logger.exception(
                "Failed to ingest paper %s: %s", paper.get("title", "unknown"), e
            )
            return None

    def update_paper_analysis(
        self,
        paper_key: str,
        summary: str | None = None,
        critique: str | None = None,
        gap_analysis: dict | None = None,
        quality_scores: dict | None = None,
    ) -> bool:
        """Update analysis data for a paper by its key (DOI, PMID, or title)."""
        sets: List[str] = []
        params: List[Any] = []
        if summary is not None:
            sets.append("summary = ?")
            params.append(summary)
        if critique is not None:
            sets.append("critique = ?")
            params.append(critique)
        if gap_analysis is not None:
            sets.append("gap_analysis = ?")
            params.append(json.dumps(gap_analysis))
        if quality_scores is not None:
            sets.append("quality_scores = ?")
            params.append(json.dumps(quality_scores))

        if not sets:
            return False

        key = str(paper_key or "")
        where = "(doi = ? OR doi_norm = ? OR pmid = ? OR title = ? OR title_norm = ?)"
        params.extend([key, _norm_doi(key), key, key, _norm_title(key)])

        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f"UPDATE papers SET {', '.join(sets)} WHERE {where}", params
                )
                self._conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            self.save_logger.exception("Failed to update paper analysis for %s: %s", paper_key, e)
            return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def filter_papers(
        self,
        topic: str | None = None,
        domain: str | None = None,
        year: str | None = None,
        doi: str | None = None,
        pmid: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[dict]:
        """Filter papers based on various criteria."""
        clauses: List[str] = []
        params: List[Any] = []

        if topic:
            clauses.append("topic_name = ?")
            params.append(topic)
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if year:
            clauses.append("year = ?")
            params.append(str(year))
        if doi:
            clauses.append("(doi = ? OR doi_norm = ?)")
            params.extend([doi, _norm_doi(doi)])
        if pmid:
            clauses.append("pmid = ?")
            params.append(str(pmid))

        sql = "SELECT * FROM papers"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        if offset is not None:
            sql += " OFFSET ?"
            params.append(int(offset))

        with self._write_lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_all_papers(self) -> List[dict]:
        """Get every paper in memory."""
        return self.filter_papers()

    def get_papers_by_topic(self, topic_name: str, limit: int | None = None) -> List[dict]:
        """Get papers by topic name."""
        return self.filter_papers(topic=topic_name, limit=limit)

    def get_paper_by_key(self, key: str) -> dict | None:
        """Get a single paper by its key (DOI, PMID, or title)."""
        key = str(key or "")
        with self._write_lock:
            row = self._conn.execute(
                "SELECT * FROM papers WHERE doi = ? OR doi_norm = ? OR pmid = ? "
                "OR title = ? OR title_norm = ? LIMIT 1",
                (key, _norm_doi(key), key, key, _norm_title(key)),
            ).fetchone()
        return self._row_to_paper(row) if row else None

    def search_papers(
        self,
        query: str,
        topic: str | None = None,
        domain: str | None = None,
        limit: int | None = None,
        sort_by: str | None = None,
        order: str = "asc",
    ) -> List[dict]:
        """Search papers by title/abstract and optional filters."""
        papers = self.filter_papers(topic=topic, domain=domain)
        query = query.lower()

        results = []
        for paper in papers:
            title = (paper.get("title", "") or "").lower()
            abstract = (paper.get("abstract", "") or "").lower()

            if not title and not abstract:
                continue

            score = 0
            if title and query in title:
                score += 100
            if abstract and query in abstract:
                score += 50
            if title:
                score += title.count(query) * 5
            if abstract:
                score += abstract.count(query)

            if score > 0:
                results.append((score, paper))

        # Sort by score or specified field
        if not sort_by:
            results.sort(key=lambda x: x[0], reverse=True)
        else:
            try:
                reverse = order.lower() == "desc"
                results.sort(
                    key=lambda x: x[1].get(sort_by, ""),
                    reverse=reverse,
                )
            except Exception as e:
                log.warning("Failed to sort by '%s': %s", sort_by, e)

        # Return only papers (without scores)
        if limit is not None:
            results = results[:limit]

        return [r[1] for r in results]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Get current memory statistics (computed live from SQLite)."""
        with self._write_lock:
            total = self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            analyzed = self._conn.execute(
                "SELECT COUNT(*) FROM papers WHERE summary IS NOT NULL AND summary != ''"
            ).fetchone()[0]
            by_topic = {
                r[0]: r[1]
                for r in self._conn.execute(
                    "SELECT topic_name, COUNT(*) FROM papers WHERE topic_name != '' GROUP BY topic_name"
                )
            }
            by_domain = {
                r[0]: r[1]
                for r in self._conn.execute(
                    "SELECT domain, COUNT(*) FROM papers WHERE domain != '' GROUP BY domain"
                )
            }

        return {
            "total_papers": total,
            "analyzed_papers": analyzed,
            "by_topic": by_topic,
            "by_domain": by_domain,
            "ids_counter": int(self.get_metadata("ids_counter", 0) or 0),
        }

    def summary(self) -> str:
        """Return a human-readable summary of the memory state."""
        stats = self.get_statistics()
        lines = [
            f"Journal Club Memory Summary ({stats.get('total_papers', 0)} papers)",
            "--------------------------------------------",
            f"  Total papers: {stats.get('total_papers', 0)}",
            f"  Topics: {len(stats.get('by_topic', {}))}",
            f"  Domains: {len(stats.get('by_domain', {}))}",
        ]

        if stats.get("by_topic"):
            lines.append("\nTopics:")
            for topic, count in sorted(stats["by_topic"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {topic} ({count} papers)")

        if stats.get("by_domain"):
            lines.append("\nDomains:")
            for domain, count in sorted(stats["by_domain"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {domain} ({count} papers)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def paper_exists(self, paper: dict) -> bool:
        """Check whether a paper (by normalized DOI, PMID, or title) is
        already stored in memory.
        """
        doi = _norm_doi(paper.get("doi"))
        pmid = str(paper.get("pmid") or "").strip()
        title = _norm_title(paper.get("title"))

        if not doi and not pmid and not title:
            return False

        with self._write_lock:
            if doi:
                row = self._conn.execute(
                    "SELECT 1 FROM papers WHERE doi_norm = ? LIMIT 1", (doi,)
                ).fetchone()
                if row:
                    return True
            if pmid:
                row = self._conn.execute(
                    "SELECT 1 FROM papers WHERE pmid = ? LIMIT 1", (pmid,)
                ).fetchone()
                if row:
                    return True
            if title:
                row = self._conn.execute(
                    "SELECT 1 FROM papers WHERE title_norm = ? LIMIT 1", (title,)
                ).fetchone()
                if row:
                    return True
        return False

    def deduplicate_papers(self) -> int:
        """Remove duplicate papers by DOI, merging analysis fields from the
        duplicates into the kept copy. Returns number of papers removed."""
        removed = 0
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT * FROM papers WHERE doi_norm != '' ORDER BY id ASC"
            ).fetchall()
            kept: Dict[str, sqlite3.Row] = {}

            for row in rows:
                doi = row["doi_norm"]
                if doi in kept:
                    kept_row = kept[doi]
                    updates: List[str] = []
                    params: List[Any] = []
                    for field in ("summary", "critique", "gap_analysis", "quality_scores"):
                        if not kept_row[field] and row[field]:
                            updates.append(f"{field} = ?")
                            params.append(row[field])
                    if updates:
                        params.append(kept_row["id"])
                        self._conn.execute(
                            f"UPDATE papers SET {', '.join(updates)} WHERE id = ?",
                            params,
                        )
                        kept[doi] = self._conn.execute(
                            "SELECT * FROM papers WHERE id = ?", (kept_row["id"],)
                        ).fetchone()
                    self._conn.execute("DELETE FROM papers WHERE id = ?", (row["id"],))
                    removed += 1
                else:
                    kept[doi] = row

            if removed:
                self._conn.commit()
                log.info("Removed %d duplicate papers by DOI", removed)

        return removed

    def remove_papers_by_doi(self, dois: List[str]) -> int:
        """Remove papers with specific DOIs from memory. Returns number removed."""
        norms = {_norm_doi(d) for d in dois}
        norms.discard("")
        if not norms:
            return 0

        placeholders = ",".join("?" * len(norms))
        with self._write_lock:
            cur = self._conn.execute(
                f"DELETE FROM papers WHERE doi_norm IN ({placeholders})",
                list(norms),
            )
            self._conn.commit()
        removed = cur.rowcount
        if removed:
            log.info("Removed %d papers by DOI", removed)
        return removed

    def scrub_thinking_traces(self) -> int:
        """Remove LLM reasoning traces (<think> blocks) from stored
        summaries and critiques. Returns the number of fields cleaned."""
        think_re = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
        cleaned = 0

        with self._write_lock:
            rows = self._conn.execute(
                "SELECT id, summary, critique FROM papers "
                "WHERE summary LIKE '%<think>%' OR critique LIKE '%<think>%'"
            ).fetchall()

            for row in rows:
                updates: List[str] = []
                params: List[Any] = []
                for field in ("summary", "critique"):
                    value = row[field]
                    if isinstance(value, str) and "<think>" in value:
                        new_value = think_re.sub("", value).strip()
                        if new_value != value:
                            updates.append(f"{field} = ?")
                            params.append(new_value)
                if updates:
                    params.append(row["id"])
                    self._conn.execute(
                        f"UPDATE papers SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
                    cleaned += len(updates)

            if cleaned:
                self._conn.commit()
                log.info("Scrubbed thinking traces from %d fields", cleaned)

        return cleaned
