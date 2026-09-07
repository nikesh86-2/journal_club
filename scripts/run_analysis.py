#!/usr/bin/env python3
"""
Run paper analysis for Journal Club.
"""
import sys
sys.path.insert(0, '.')
import argparse
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
from core.literature_memory import JournalClubMemory
from core.paper_analyzer import analyze_batch
from core.config import get_topic_by_name


def main():
    parser = argparse.ArgumentParser(description="Run paper analysis for Journal Club.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-analyze every paper (default: only papers without a summary)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max papers per topic (default: 50)",
    )
    args = parser.parse_args()

    memory = JournalClubMemory()
    stats = memory.get_statistics()
    print(f'Total papers in memory: {stats.get("total_papers", 0)}')

    for topic in stats['by_topic'].keys():
        papers = memory.filter_papers(topic=topic, limit=args.limit)
        to_analyze = papers if args.all else [p for p in papers if not p.get('summary')]
        if not to_analyze:
            print(f'No papers to analyze for topic: {topic}')
            continue

        # Use the domain configured for this topic, falling back to the
        # domain stored on the papers themselves, then 'general'.
        topic_conf = get_topic_by_name(topic)
        domain = (
            (topic_conf.get("domain") if topic_conf else None)
            or to_analyze[0].get("domain")
            or "general"
        )

        print(f'Analyzing {len(to_analyze)} papers for topic: {topic} (domain: {domain})')
        results = analyze_batch(to_analyze, domain=domain, memory=memory)
        for paper, result in zip(to_analyze, results):
            # Update memory with analysis
            key = paper.get('doi') or paper.get('pmid') or paper.get('title')
            memory.update_paper_analysis(
                key,
                summary=result.get('summary'),
                critique=result.get('critique'),
                gap_analysis=result.get('gap_analysis'),
                quality_scores=result.get('quality_scores'),
            )

    print("Analysis complete!")


if __name__ == "__main__":
    main()
