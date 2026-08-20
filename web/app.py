"""
app.py

Flask web interface for Journal Club.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, render_template, jsonify, send_file

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from core.literature_memory import JournalClubMemory
from core.report_generator import (
    generate_topic_report,
    generate_paper_report,
    generate_summary_report,
)

log = logging.getLogger("journal_club.web")

app = Flask(__name__)

# Configuration
MEMORY_PATH = os.getenv("JOURNAL_CLUB_LITERATURE_MEMORY_PATH", "literature_memory.json")
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "markdown"


def get_memory():
    """Get or create memory instance."""
    return JournalClubMemory(MEMORY_PATH)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Topic dashboard."""
    memory = get_memory()
    stats = memory.get_statistics()
    
    return render_template("index.html", stats=stats)


@app.route("/topic/<topic_name>")
def topic_detail(topic_name):
    """Topic detail view with papers."""
    memory = get_memory()
    papers = memory.filter_papers(topic=topic_name, limit=100)
    
    return render_template(
        "topic_view.html",
        topic_name=topic_name,
        papers=papers,
        paper_count=len(papers),
    )


@app.route("/paper/<paper_key>")
def paper_detail(paper_key):
    """Individual paper with full analysis."""
    memory = get_memory()
    paper = memory.get_paper_by_key(paper_key)
    
    if not paper:
        return "Paper not found", 404
    
    return render_template("paper_detail.html", paper=paper)


@app.route("/api/stats")
def api_stats():
    """API endpoint for statistics."""
    memory = get_memory()
    stats = memory.get_statistics()
    return jsonify(stats)


@app.route("/api/topics")
def api_topics():
    """API endpoint for topics."""
    memory = get_memory()
    stats = memory.get_statistics()
    return jsonify(list(stats["by_topic"].keys()))


@app.route("/api/papers/<topic_name>")
def api_papers(topic_name):
    """API endpoint for papers by topic."""
    memory = get_memory()
    papers = memory.filter_papers(topic=topic_name, limit=100)
    return jsonify(papers)


@app.route("/api/trigger-analysis", methods=["POST"])
def api_trigger_analysis():
    """Trigger paper analysis via API."""
    try:
        from core.paper_analyzer import analyze_batch
        memory = get_memory()
        stats = memory.get_statistics()
        analyzed_count = 0
        for topic in stats['by_topic'].keys():
            papers = memory.filter_papers(topic=topic, limit=20)
            unanalyzed = [p for p in papers if not p.get('summary')]
            if unanalyzed:
                results = analyze_batch(unanalyzed, "general")
                for result in results:
                    if result.get('analysis'):
                        paper = result['paper']
                        analysis = result['analysis']
                        key = paper.get('doi') or paper.get('title')
                        memory.update_paper_analysis(
                            key,
                            summary=analysis.get('summary'),
                            critique=analysis.get('critique'),
                            gap_analysis=analysis.get('gap_analysis'),
                            quality_scores=analysis.get('quality_scores'),
                        )
                        analyzed_count += 1
        return jsonify({"status": "success", "analyzed_papers": analyzed_count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/export/<topic_name>/markdown")
def export_markdown(topic_name):
    """Download markdown report for topic."""
    memory = get_memory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        report_path = generate_topic_report(topic_name, memory, str(OUTPUT_DIR))
        return send_file(report_path, as_attachment=True, download_name=f"{topic_name}_report.md")
    except Exception as e:
        return f"Error generating report: {e}", 500


@app.route("/export/<topic_name>/json")
def export_json(topic_name):
    """Download JSON export for topic."""
    memory = get_memory()
    papers = memory.filter_papers(topic=topic_name, limit=100)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    import json
    from datetime import datetime
    
    export_data = {
        "topic": topic_name,
        "exported_at": datetime.utcnow().isoformat(),
        "paper_count": len(papers),
        "papers": papers,
    }
    
    json_path = OUTPUT_DIR / f"{topic_name}_export.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return send_file(json_path, as_attachment=True, download_name=f"{topic_name}_export.json")


@app.route("/export/summary/markdown")
def export_summary():
    """Download summary markdown report."""
    memory = get_memory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        report_path = generate_summary_report(memory, str(OUTPUT_DIR))
        return send_file(report_path, as_attachment=True, download_name="summary_report.md")
    except Exception as e:
        return f"Error generating report: {e}", 500


def run_server(host="127.0.0.1", port=None, debug=False):
    """Run the Flask server."""
    if port is None:
        port = int(os.getenv("JOURNAL_CLUB_WEB_PORT", "5000"))
    
    log.info("Starting Journal Club web server on %s:%d", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
