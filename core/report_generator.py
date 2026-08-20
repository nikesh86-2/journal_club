"""
report_generator.py

Markdown report generation for Journal Club.

Responsibilities:
  - Generate markdown reports for topics
  - Generate individual paper reports
  - Generate summary statistics
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .literature_memory import JournalClubMemory

log = logging.getLogger("journal_club.reports")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_topic_report(
    topic_name: str,
    memory: JournalClubMemory,
    output_dir: str | None = None,
) -> str:
    """Generate a markdown report for a topic."""

    papers = memory.get_papers_by_topic(topic_name, limit=100)

    if not papers:
        return f"# Topic Report: {topic_name}\n\nNo papers found for this topic.\n"

    if output_dir is None:
        output_dir = Path(__file__).parents[1] / "output" / "markdown"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build report
    report_lines = [
        f"# Journal Club Report: {topic_name}",
        f"",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Total Papers:** {len(papers)}",
        f"",
    ]

    # Add statistics
    analyzed = sum(1 for p in papers if p.get("summary"))
    report_lines.extend([
        f"## Statistics",
        f"",
        f"- Papers analyzed: {analyzed}/{len(papers)}",
        f"- Papers with gap analysis: {sum(1 for p in papers if p.get('gap_analysis'))}",
        f"- Papers with quality scores: {sum(1 for p in papers if p.get('quality_scores'))}",
        f"",
    ])

    # Add papers
    report_lines.extend([
        f"## Papers",
        f"",
    ])

    for i, paper in enumerate(papers, 1):
        report_lines.extend(_format_paper_section(paper, i))

    # Write report
    report_content = "\n".join(report_lines)

    filename = output_dir / f"{topic_name.replace(' ', '_').lower()}_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    log.info("Generated topic report: %s", filename)
    return str(filename)


def _format_paper_section(paper: Dict[str, Any], index: int) -> List[str]:
    """Format a paper section for the report."""

    lines = [
        f"### {index}. {paper.get('title', 'Untitled')}",
        f"",
    ]

    # Basic metadata
    if paper.get("authors"):
        lines.append(f"**Authors:** {paper.get('authors')}")

    if paper.get("year"):
        lines.append(f"**Year:** {paper.get('year')}")

    if paper.get("doi"):
        lines.append(f"**DOI:** [{paper.get('doi')}](https://doi.org/{paper.get('doi')})")

    if paper.get("url"):
        lines.append(f"**URL:** {paper.get('url')}")

    lines.append("")

    # Summary
    if paper.get("summary"):
        lines.extend([
            f"#### Summary",
            f"",
            paper.get("summary"),
            f"",
        ])

    # Gap analysis
    gap_analysis = paper.get("gap_analysis", {})
    if any(gap_analysis.values()):
        lines.extend([
            f"#### Gap Analysis",
            f"",
        ])

        for category, gaps in gap_analysis.items():
            if gaps:
                lines.append(f"**{category.capitalize()}:**")
                for gap in gaps:
                    lines.append(f"- {gap}")
                lines.append("")

    # Quality scores
    quality_scores = paper.get("quality_scores", {})
    if quality_scores and any(v is not None for v in quality_scores.values()):
        lines.extend([
            f"#### Quality Scores",
            f"",
        ])

        for metric, score in quality_scores.items():
            if score is not None:
                lines.append(f"- **{metric.replace('_', ' ').title()}:** {score}/1.0")

        lines.append("")

    # Critique
    if paper.get("critique"):
        lines.extend([
            f"#### Critique",
            f"",
            paper.get("critique"),
            f"",
        ])

    # Recommendations
    if paper.get("recommendation_type"):
        lines.extend([
            f"#### Recommendation Type",
            f"",
            f"{paper.get('recommendation_type').title()}",
            f"",
        ])

    if paper.get("related_papers"):
        lines.extend([
            f"#### Related Papers",
            f"",
        ])
        for related in paper.get("related_papers", [])[:5]:
            if isinstance(related, dict):
                title = related.get("title", "Unknown")
                lines.append(f"- {title}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def generate_paper_report(
    paper: Dict[str, Any],
    output_dir: str | None = None,
) -> str:
    """Generate a detailed markdown report for a single paper."""

    if output_dir is None:
        output_dir = Path(__file__).parents[1] / "output" / "markdown"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build report
    report_lines = [
        f"# Paper Report: {paper.get('title', 'Untitled')}",
        f"",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"",
    ]

    # Metadata
    report_lines.extend([
        f"## Metadata",
        f"",
    ])

    metadata_fields = [
        ("Title", "title"),
        ("Authors", "authors"),
        ("Year", "year"),
        ("DOI", "doi"),
        ("PMID", "pmid"),
        ("URL", "url"),
        ("Publication Date", "publication_date"),
        ("Topic", "topic_name"),
        ("Domain", "domain"),
    ]

    for label, field in metadata_fields:
        value = paper.get(field)
        if value:
            if field == "doi":
                report_lines.append(f"**{label}:** [{value}](https://doi.org/{value})")
            elif field == "url":
                report_lines.append(f"**{label}:** {value}")
            else:
                report_lines.append(f"**{label}:** {value}")

    report_lines.append("")

    # Abstract
    if paper.get("abstract"):
        report_lines.extend([
            f"## Abstract",
            f"",
            paper.get("abstract"),
            f"",
        ])

    # Summary
    if paper.get("summary"):
        report_lines.extend([
            f"## Summary",
            f"",
            paper.get("summary"),
            f"",
        ])

    # Gap analysis
    gap_analysis = paper.get("gap_analysis", {})
    if any(gap_analysis.values()):
        report_lines.extend([
            f"## Gap Analysis",
            f"",
        ])

        for category, gaps in gap_analysis.items():
            if gaps:
                report_lines.append(f"### {category.capitalize()}")
                for gap in gaps:
                    report_lines.append(f"- {gap}")
                report_lines.append("")

    # Quality scores
    quality_scores = paper.get("quality_scores", {})
    if quality_scores and any(v is not None for v in quality_scores.values()):
        report_lines.extend([
            f"## Quality Scores",
            f"",
        ])

        for metric, score in quality_scores.items():
            if score is not None:
                # Add visual indicator
                if score >= 0.8:
                    indicator = "🟢"
                elif score >= 0.6:
                    indicator = "🟡"
                else:
                    indicator = "🔴"

                report_lines.append(f"- {indicator} **{metric.replace('_', ' ').title()}:** {score}/1.0")

        report_lines.append("")

    # Critique
    if paper.get("critique"):
        report_lines.extend([
            f"## Critique",
            f"",
            paper.get("critique"),
            f"",
        ])

    # Recommendations
    if paper.get("recommendation_type"):
        report_lines.extend([
            f"## Recommendation Type",
            f"",
            f"**{paper.get('recommendation_type').title()}**",
            f"",
        ])

    if paper.get("related_papers"):
        report_lines.extend([
            f"## Related Papers",
            f"",
        ])
        for related in paper.get("related_papers", []):
            if isinstance(related, dict):
                title = related.get("title", "Unknown")
                doi = related.get("doi", "")
                if doi:
                    report_lines.append(f"- [{title}](https://doi.org/{doi})")
                else:
                    report_lines.append(f"- {title}")
        report_lines.append("")

    # Write report
    report_content = "\n".join(report_lines)

    # Generate filename from DOI or title
    if paper.get("doi"):
        filename = output_dir / f"paper_{paper.get('doi').replace('/', '_')}.md"
    else:
        safe_title = "".join(c for c in paper.get("title", "") if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = output_dir / f"paper_{safe_title.replace(' ', '_')[:50]}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    log.info("Generated paper report: %s", filename)
    return str(filename)


def generate_summary_report(
    memory: JournalClubMemory,
    output_dir: str | None = None,
) -> str:
    """Generate a summary statistics report."""

    stats = memory.get_statistics()

    if output_dir is None:
        output_dir = Path(__file__).parents[1] / "output" / "markdown"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build report
    report_lines = [
        f"# Journal Club Summary Report",
        f"",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"",
        f"## Overall Statistics",
        f"",
        f"- **Total Papers:** {stats['total_papers']}",
        f"- **Analyzed Papers:** {stats['analyzed_papers']}",
        f"- **Topics Tracked:** {len(stats['by_topic'])}",
        f"- **Domains Tracked:** {len(stats['by_domain'])}",
        f"",
    ]

    # By topic
    if stats["by_topic"]:
        report_lines.extend([
            f"## Papers by Topic",
            f"",
        ])
        for topic, count in sorted(stats["by_topic"].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- **{topic}:** {count} papers")
        report_lines.append("")

    # By domain
    if stats["by_domain"]:
        report_lines.extend([
            f"## Papers by Domain",
            f"",
        ])
        for domain, count in sorted(stats["by_domain"].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- **{domain}:** {count} papers")
        report_lines.append("")

    # Write report
    report_content = "\n".join(report_lines)

    filename = output_dir / "summary_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    log.info("Generated summary report: %s", filename)
    return str(filename)


def generate_all_reports(
    memory: JournalClubMemory,
    output_dir: str | None = None,
) -> Dict[str, str]:
    """Generate all reports (summary + topic reports)."""

    generated = {}

    # Summary report
    generated["summary"] = generate_summary_report(memory, output_dir)

    # Get all topics
    stats = memory.get_statistics()
    for topic_name in stats["by_topic"].keys():
        try:
            topic_report = generate_topic_report(topic_name, memory, output_dir)
            generated[topic_name] = topic_report
        except Exception as e:
            log.warning("Failed to generate report for topic %s: %s", topic_name, e)

    return generated
