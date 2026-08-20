#!/usr/bin/env python3
"""
Simple analyzer worker that loads the local LLM model in a separate process
and runs analysis over papers stored in the memory cache. This isolates model
loading from the streaming threads to avoid OOM kills.

Usage:
    python3 scripts/analyzer_worker.py [--memory-file cache/journal_club_memory.json] [--output-dir results/analysis]

The script will process every paper in the memory file and write per-paper
JSON results to the output directory.
"""

import argparse
import json
import logging
import sys
import re
from pathlib import Path
import hashlib
import time

# Force CPU offload by default for worker stability
import os
os.environ.setdefault("JOURNAL_CLUB_FORCE_CPU_OFFLOAD", "1")

# Ensure repo root (bot/journal_club) is on sys.path so `from core...` works
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from core.paper_analyzer import analyze_paper

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
log = logging.getLogger("journal_club.analyzer.worker")


def slugify(text: str) -> str:
    h = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
    return re.sub(r'[^a-z0-9]+', '-', text.lower())[:40] + "-" + h


def load_memory(path: Path):
    if not path.exists():
        log.error("Memory file not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text())
        # Support a few common structures
        if isinstance(data, dict):
            if "papers" in data and isinstance(data["papers"], list):
                return data["papers"]
            # maybe list of values
            vals = [v for v in data.values() if isinstance(v, dict) and "title" in v]
            if vals:
                return vals
        if isinstance(data, list):
            return data
    except Exception as e:
        log.error("Failed to parse memory file: %s", e)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-file", default="cache/journal_club_memory.json")
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()

    mem_path = Path(args.memory_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = load_memory(mem_path)
    log.info("Loaded %d papers from memory: %s", len(papers), mem_path)

    for i, paper in enumerate(papers):
        title = paper.get("title") or paper.get("doi") or f"paper-{i}"
        safe_name = hashlib.sha1(title.encode('utf-8')).hexdigest()[:10]
        out_file = out_dir / f"analysis-{safe_name}.json"
        if out_file.exists():
            log.info("Skipping already-analyzed paper: %s", title)
            continue
        try:
            log.info("Analyzing paper %d/%d: %s", i + 1, len(papers), title)
            result = analyze_paper(paper)
            out_file.write_text(json.dumps(result, indent=2))
            log.info("Wrote result: %s", out_file)
        except Exception as e:
            log.exception("Failed to analyze paper: %s", e)

    log.info("Worker run complete")


if __name__ == '__main__':
    import re
    main()
