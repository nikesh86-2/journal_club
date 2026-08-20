"""
literature_memory.py

Memory backend for Journal Club.

Responsibilities:
  - Store and manage academic papers in memory
  - Provide methods to query, filter, and aggregate papers
  - Persist memory state to disk
  - Support for indexing and search
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import tempfile
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, Dict, List

from . import config

log = logging.getLogger("journal_club.memory")

# Set up quiet time handler
_quiet_time = datetime(datetime.utcnow().year, datetime.utcnow().month, datetime.utcnow().day, 0, 0, 0, 0)

class JournalClubMemory:
    """Memory backend for Journal Club."""

    def __init__(self, path: str | None = None):
        """Initialize memory, optionally from a file."""
        self.path = path or self._default_path()
        
        # Initialize loggers first so they're always available
        self.save_logger = logging.getLogger("journal_club.memory.save")
        self.stat_logger = logging.getLogger("journal_club.memory.stats")

        if os.path.exists(self.path):
            self.memory = self._load()
        else:
            self.memory = {
                "papers": [],
                "schema_version": "journal_club_memory.v1",
                "statistics": {
                    "total_papers": 0,
                    "by_topic": {},
                    "by_domain": {},
                    "ids_counter": 0,
                },
                "updated_cache": {}
            }
            self.save()

    def _default_path(self) -> str:
        """Get default path for memory file."""
        return str(config.CACHE_DIR / "journal_club_memory.json")

    def _load(self) -> dict:
        """Load memory from file if it exists."""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            log.warning("Memory file not found or empty at %s, initializing empty", self.path)
            return {
                "papers": [],
                "schema_version": "journal_club_memory.v1",
                "statistics": {
                    "total_papers": 0,
                    "by_topic": {},
                    "by_domain": {},
                    "ids_counter": 0,
                },
                "updated_cache": {}
            }

        try:
            # Use utf-8-sig to handle BOM if present
            with open(self.path, "r", encoding="utf-8-sig") as f:
                memory = json.load(f)

            if memory.get("schema_version") != "journal_club_memory.v1":
                log.warning("Memory has old schema version: %s", memory.get("schema_version"))
                self._migrate_schema(memory)

            return memory

        except Exception as e:
            log.error("Failed to load memory from %s: %s", self.path, e)
            return {
                "papers": [],
                "schema_version": "journal_club_memory.v1",
                "statistics": {
                    "total_papers": 0,
                    "by_topic": {},
                    "by_domain": {},
                    "ids_counter": 0,
                },
                "updated_cache": {}
            }

    def _migrate_schema(self, memory: dict) -> None:
        """Migrate memory from old schema to current version."""
        log.warning("Migrating memory schema...")

        memory["schema_version"] = "journal_club_memory.v1"
        memory.setdefault("papers", [])
        memory.setdefault("statistics", {
            "total_papers": len(memory.get("papers", [])),
            "by_topic": {},
            "by_domain": {},
            "ids_counter": 0,
        })
        memory.setdefault("updated_cache", {})

        log.warning("Memory migrated to schema version: journal_club_memory.v1")

    def _save(self) -> None:
        """
        Attempt to atomically save the memory to the filesystem.
        """
        try:
            self._compact()
            self.memory["schema_version"] = "journal_club_memory.v1"
            dir_name = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(dir_name, exist_ok=True)

            # Thread-safe tmp write
            with tempfile.NamedTemporaryFile(
                "w",
                dir=dir_name,
                delete=False,
                encoding="utf-8",
                errors="replace"
            ) as tf:
                # Custom JSON encoder to handle datetime/time objects
                json.dump(self.memory, tf, indent=2, ensure_ascii=False, default=self._json_serializer)
                temp_name = tf.name

            os.replace(temp_name, self.path)

            # Log memory size for monitoring
            self.stat_logger.info(f"Successfully saved memory: {self.path}, papers: {len(self.memory.get('papers', []))}")

        except Exception:
            self.save_logger.exception(f"Failed to save memory to {self.path}")
            raise

    def save(self) -> None:
        """Save memory to file."""
        try:
            self._save()
        except Exception as e:
            self.save_logger.exception(e)
            raise

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (datetime, time, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _compact(self) -> None:
        """Compact the memory by removing incomplete records."""
        compacts = {
            key: value for key, value in self.memory.items() if key not in ("updated_cache")
        }
        compacts["updated_cache"] = {}

        # Remove invalid papers
        compacts["papers"] = [
            paper for paper in self.memory.get("papers", [])
            if all([
                paper.get("title"),
                paper.get("schema_version"),
                isinstance(paper.get("timestamp"), (str, datetime, time, date)),
                paper.get("topic_name")
            ])
        ]
        compacts["papers"].sort(key=lambda x: str(x.get("timestamp", 0)), reverse=False)

        # Recalculate statistics
        self.update_statistics(compacts)

    def update_statistics(self, memory: dict | None = None) -> None:
        """Update statistics for memory."""
        mem = memory or self.memory

        by_topic = {}
        by_domain = {}
        total_papers = len(mem.get("papers", []))
        topics_counter = 0
        domains_counter = 0

        for paper in mem.get("papers", []):
            topic = paper.get("topic_name")
            domain = paper.get("domain")

            if topic and domain:
                by_topic[topic] = by_topic.get(topic, 0) + 1
                by_domain[domain] = by_domain.get(domain, 0) + 1

        mem["statistics"] = {
            "total_papers": total_papers,
            "by_topic": by_topic,
            "by_domain": by_domain,
            "ids_counter": mem["statistics"].get("ids_counter", 0),
        }

        if memory is None:
            self.memory = mem

    def ingest_paper(
        self,
        paper: dict | Any,
        topic_name: str | None = None,
        domain: str | None = None,
        time_window_months: int = 12,
    ) -> dict | None:
        """
        Ingest a paper into memory with its metadata.
        """
        try:
            # Extract paper metadata
            title = paper.get("title", "untitled")
            abstract = paper.get("abstract", "")
            year = paper.get("year", "")
            pub_date = paper.get("publication_date", "")
            doi = paper.get("doi", "")
            pmid = paper.get("pmid", "")
            url = paper.get("url", "")
            source = paper.get("source", "")
            authors = paper.get("authors", [])

            # Safely clean entry record
            research_metadata = {
                "schema_version": paper.get("schema_version", "journal_club_paper.v1"),
                "timestamp": _quiet_time.isoformat(),
                "title": title,
                "abstract": abstract,
                "year": year,
                "publication_date": pub_date,
                "doi": doi,
                "pmid": pmid,
                "url": url,
                "source": source,
                "authors": authors,
                "topic_name": topic_name,
                "domain": domain,
            }

            # Inject the actual analysis result as soon as it's obtained, for quicker updates
            if 'summary' in paper: research_metadata['summary'] = paper['summary']
            if 'gap_analysis' in paper: research_metadata['gap_analysis'] = paper['gap_analysis']
            if 'quality_scores' in paper: research_metadata['quality_scores'] = paper['quality_scores']

            self.memory["papers"].append(research_metadata)
            self.update_statistics()
            self.save()
            self.save_logger.info(f"Successfully appended paper {research_metadata['title'][:60]}")

            return research_metadata
        except Exception as e:
            self.save_logger.exception(f"Failed to ingest paper {paper.get('title', 'unknown')}: {str(e)}")

    def update_paper_analysis(
        self,
        paper_key: str,
        summary: str | None = None,
        critique: str | None = None,
        gap_analysis: dict | None = None,
        quality_scores: dict | None = None,
    ) -> bool:
        """
        Update analysis data for a paper by its key (DOI, PMID, or title).
        """
        try:
            updated = False

            for paper in self.memory.get("papers", []):
                if paper.get("doi") == paper_key or paper.get("pmid") == paper_key or paper.get("title") == paper_key:
                    if summary is not None: paper["summary"] = summary
                    if critique is not None: paper["critique"] = critique
                    if gap_analysis is not None: paper["gap_analysis"] = gap_analysis
                    if quality_scores is not None: paper["quality_scores"] = quality_scores

                    updated = True
                    break

            if updated:
                self.update_statistics()
                return True

            return False
        except Exception as e:
            self.save_logger.exception(f"Failed to update paper analysis for {paper_key}: {str(e)}")

    def filter_papers(
        self,
        topic: str | None = None,
        domain: str | None = None,
        year: str | None = None,
        doi: str | None = None,
        pmid: str | None = None,
        limit: int | None = None,
    ) -> List[dict]:
        """
        Filter papers based on various criteria.
        """
        papers = self.memory.get("papers", [])

        if topic: papers = [p for p in papers if p.get("topic_name") == topic]
        if domain: papers = [p for p in papers if p.get("domain") == domain]
        if year: papers = [p for p in papers if str(p.get("year")) == year]
        if doi: papers = [p for p in papers if p.get("doi") == doi]
        if pmid: papers = [p for p in papers if p.get("pmid") == pmid]

        if limit is not None: papers = papers[:limit]

        return papers

    def get_papers_by_topic(self, topic_name: str, limit: int | None = None) -> List[dict]:
        """Get papers by topic name."""
        return self.filter_papers(topic=topic_name, limit=limit)

    def get_paper_by_key(self, key: str) -> dict | None:
        """Get a single paper by its key (DOI, PMID, or title)."""
        for paper in self.memory.get("papers", []):
            if paper.get("doi") == key or paper.get("pmid") == key or paper.get("title") == key:
                return paper
        return None

    def search_papers(
        self,
        query: str,
        topic: str | None = None,
        domain: str | None = None,
        limit: int | None = None,
        sort_by: str | None = None,
        order: str = "asc",
    ) -> List[dict]:
        """
        Search papers by title/abstract and optional filters.
        """
        papers = self.filter_papers(topic=topic, domain=domain)
        query = query.lower()

        results = []
        for paper in papers:
            title = (paper.get("title", "") or "").lower()
            abstract = (paper.get("abstract", "") or "").lower()

            if not title and not abstract:
                continue

            score = 0
            if title and query in title: score += 100
            if abstract and query in abstract: score += 50
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

    def get_statistics(self) -> dict:
        """Get current memory statistics."""
        mem = self.memory

        if not mem.get("statistics"):
            return {
                "total_papers": len(mem.get("papers", [])),
                "by_topic": {},
                "by_domain": {},
                "ids_counter": 0,
            }

        return mem.get("statistics", {})

    def summary(self) -> str:
        """Return a human-readable summary of the memory state."""
        stats = self.get_statistics()
        lines = [
            f"Journal Club Memory Summary ({len(stats.get('papers', []))} papers)",
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