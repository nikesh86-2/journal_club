#!/usr/bin/env python3
"""
Backfill citation counts for papers already in memory using the
Semantic Scholar batch endpoint (one request per 100 DOIs).

Usage:
    python scripts/backfill_citations.py [--only-missing]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

from core.literature_memory import JournalClubMemory
from core.research_agent_adaptive import fetch_citation_counts_by_doi


def main():
    parser = argparse.ArgumentParser(description="Backfill citation counts from Semantic Scholar.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Only fetch for papers without a citation count (default: on)",
    )
    parser.add_argument(
        "--all",
        dest="only_missing",
        action="store_false",
        help="Refresh citation counts for every paper with a DOI",
    )
    args = parser.parse_args()

    memory = JournalClubMemory()
    papers = memory.get_all_papers()
    targets = [
        p for p in papers
        if p.get("doi") and (not args.only_missing or not p.get("citation_count"))
    ]

    if not targets:
        print("No papers need citation counts.")
        return

    print(f"Fetching citation counts for {len(targets)} papers...")
    counts = fetch_citation_counts_by_doi([p["doi"] for p in targets])

    updated = 0
    for p in targets:
        count = counts.get(str(p["doi"]).strip().lower())
        if count is not None:
            p["citation_count"] = count
            updated += 1

    print(f"Updated {updated}/{len(targets)} papers")
    if updated:
        memory.update_statistics()
        memory.save()


if __name__ == "__main__":
    main()
