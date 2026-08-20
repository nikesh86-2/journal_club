#!/usr/bin/env python3
"""
Run paper analysis for Journal Club.
"""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
from core.literature_memory import JournalClubMemory
from core.paper_analyzer import analyze_batch

memory = JournalClubMemory()
stats = memory.get_statistics()
print(f'Total papers in memory: {stats.get("total_papers", 0)}')

for topic in stats['by_topic'].keys():
    papers = memory.filter_papers(topic=topic, limit=50)
    unanalyzed = [p for p in papers if not p.get('summary')]
    if unanalyzed:
        print(f'Analyzing {len(unanalyzed)} papers for topic: {topic}')
        results = analyze_batch(unanalyzed, domain='general')
        for i, result in enumerate(results):
            if i < len(unanalyzed):
                paper = unanalyzed[i]
                # Update memory with analysis
                key = paper.get('doi') or paper.get('title')
                memory.update_paper_analysis(
                    key,
                    summary=result.get('summary'),
                    critique=result.get('critique'),
                    gap_analysis=result.get('gap_analysis'),
                    quality_scores=result.get('quality_scores'),
                )
    else:
        print(f'No unanalyzed papers for topic: {topic}')

print("Analysis complete!")