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
MEMORY_PATH = os.getenv("JOURNAL_CLUB_LITERATURE_MEMORY_PATH")
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


@app.route("/paper/<path:paper_key>")
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


@app.route("/api/papers")
def api_papers_all():
    """API endpoint for papers with query parameters."""
    from flask import request
    memory = get_memory()
    topic = request.args.get("topic")
    domain = request.args.get("domain")
    limit = request.args.get("limit", 100, type=int)
    papers = memory.filter_papers(topic=topic, domain=domain, limit=limit)
    return jsonify(papers)


@app.route("/api/papers/<topic_name>")
def api_papers(topic_name):
    """API endpoint for papers by topic."""
    memory = get_memory()
    papers = memory.filter_papers(topic=topic_name, limit=100)
    return jsonify(papers)


@app.route("/api/paper/<path:paper_key>")
def api_paper_detail(paper_key):
    """API endpoint for single paper details."""
    memory = get_memory()
    paper = memory.get_paper_by_key(paper_key)
    if not paper:
        return jsonify({"error": "Paper not found"}), 404
    return jsonify(paper)


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
                topic_conf = get_topic_config_by_name(topic)
                topic_domain = topic_conf.get("domain", "general") if topic_conf else "general"
                results = analyze_batch(unanalyzed, topic_domain)
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


# ---------------------------------------------------------------------------
# Dynamic Configuration & Ingestion API Endpoints
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parents[1] / "config"


def get_topic_config_by_name(name: str):
    import yaml
    topics_path = CONFIG_DIR / "topics.yaml"
    if topics_path.exists():
        with open(topics_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            for t in data.get("topics", []):
                if t.get("name") == name:
                    return t
    return None


@app.route("/api/refresh-stats", methods=["POST"])
def api_refresh_stats():
    """Force refresh statistics from current papers."""
    try:
        memory = get_memory()
        memory.update_statistics()
        memory.save()
        stats = memory.get_statistics()
        return jsonify({"status": "success", "stats": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config/sources", methods=["GET", "POST"])
def api_config_sources():
    """Get or update API and LLM settings."""
    import yaml
    from flask import request

    settings_path = CONFIG_DIR / "settings.yaml"

    if request.method == "POST":
        payload = request.get_json() or {}
        current_settings = {}
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                current_settings = yaml.safe_load(f) or {}

        if "llm_server_url" in payload:
            os.environ["JOURNAL_CLUB_LLAMA_SERVER_URL"] = str(payload["llm_server_url"]).strip()
        if "llm_temperature" in payload:
            try:
                current_settings["llm_temperature"] = float(payload["llm_temperature"])
                os.environ["JOURNAL_CLUB_LLM_TEMPERATURE"] = str(payload["llm_temperature"])
            except ValueError:
                pass
        if "llm_max_tokens" in payload:
            try:
                current_settings["llm_max_tokens"] = int(payload["llm_max_tokens"])
                os.environ["JOURNAL_CLUB_LLM_MAX_TOKENS"] = str(payload["llm_max_tokens"])
            except ValueError:
                pass
        if "default_time_window_months" in payload:
            try:
                current_settings["default_time_window_months"] = int(payload["default_time_window_months"])
                os.environ["JOURNAL_CLUB_TIME_WINDOW_MONTHS"] = str(payload["default_time_window_months"])
            except ValueError:
                pass
        if "stream_batch_size" in payload:
            try:
                current_settings["stream_batch_size"] = int(payload["stream_batch_size"])
                os.environ["JOURNAL_CLUB_STREAM_BATCH_SIZE"] = str(payload["stream_batch_size"])
            except ValueError:
                pass
        if "active_search_api" in payload:
            current_settings["active_search_api"] = str(payload["active_search_api"])

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(current_settings, f, default_flow_style=False, sort_keys=False)

        return jsonify({"status": "success", "message": "API settings updated successfully", "settings": current_settings})

    # GET
    settings = {}
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}

    response_data = {
        "active_search_api": settings.get("active_search_api", "Europe PMC (Preprints)"),
        "available_search_apis": [
            {"id": "europe_pmc", "name": "Europe PMC (Preprints)", "status": "Active / Primary", "supported": True},
            {"id": "biorxiv", "name": "BioRxiv / MedRxiv Direct", "status": "Integrated via Europe PMC", "supported": True},
            {"id": "semantic_scholar", "name": "Semantic Scholar API", "status": "Available", "supported": True},
            {"id": "ncbi_entrez", "name": "NCBI PubMed / Entrez", "status": "Available", "supported": True},
        ],
        "llm_server_url": os.getenv("JOURNAL_CLUB_LLAMA_SERVER_URL", "http://localhost:8080"),
        "llm_temperature": settings.get("llm_temperature", float(os.getenv("JOURNAL_CLUB_LLM_TEMPERATURE", 0.3))),
        "llm_max_tokens": settings.get("llm_max_tokens", int(os.getenv("JOURNAL_CLUB_LLM_MAX_TOKENS", 2000))),
        "time_window_months": settings.get("default_time_window_months", int(os.getenv("JOURNAL_CLUB_TIME_WINDOW_MONTHS", 240))),
        "stream_batch_size": settings.get("stream_batch_size", int(os.getenv("JOURNAL_CLUB_STREAM_BATCH_SIZE", 20))),
    }
    return jsonify(response_data)


@app.route("/api/config/test-source", methods=["POST"])
def api_test_source():
    """Test connection to search API or LLM inference endpoint."""
    from flask import request
    import urllib.request
    import json
    payload = request.get_json() or {}
    target = payload.get("target", "all")

    results = {}

    # Test LLM endpoint
    if target in ("all", "llm"):
        server_url = payload.get("llm_server_url") or os.getenv("JOURNAL_CLUB_LLAMA_SERVER_URL", "http://localhost:8080")
        try:
            req = urllib.request.Request(f"{server_url}/health", headers={"User-Agent": "JournalClub/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                status = resp.getcode()
                results["llm"] = {"status": "connected", "url": server_url, "http_code": status, "message": "LLM Inference Server healthy"}
        except Exception as e:
            results["llm"] = {"status": "unavailable", "url": server_url, "error": str(e), "message": "Could not connect to LLM server"}

    # Test Europe PMC search
    if target in ("all", "europe_pmc", "search"):
        try:
            test_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=SRC:PPR&format=json&pageSize=1"
            req = urllib.request.Request(test_url, headers={"User-Agent": "JournalClub/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                hit_count = data.get("hitCount", 0)
                results["search_api"] = {"status": "connected", "provider": "Europe PMC", "hit_count": hit_count, "message": "Search API responsive"}
        except Exception as e:
            results["search_api"] = {"status": "unavailable", "provider": "Europe PMC", "error": str(e)}

    return jsonify(results)


@app.route("/api/config/topics", methods=["GET", "POST"])
def api_config_topics():
    """Get all topics or add/update a topic."""
    import yaml
    from flask import request

    topics_path = CONFIG_DIR / "topics.yaml"
    data = {"topics": []}
    if topics_path.exists():
        with open(topics_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {"topics": []}

    if request.method == "POST":
        topic_payload = request.get_json() or {}
        name = topic_payload.get("name", "").strip()
        if not name:
            return jsonify({"status": "error", "message": "Topic name is required"}), 400

        # Check if topic already exists to update it, else append
        existing = next((t for t in data["topics"] if t.get("name") == name), None)
        if existing:
            existing.update(topic_payload)
        else:
            data["topics"].append(topic_payload)

        with open(topics_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return jsonify({"status": "success", "message": f"Topic '{name}' saved successfully", "topic": topic_payload})

    return jsonify(data.get("topics", []))


@app.route("/api/config/topics/<path:topic_name>", methods=["DELETE"])
def api_delete_topic(topic_name):
    """Delete a topic from topics.yaml."""
    import yaml
    topics_path = CONFIG_DIR / "topics.yaml"
    if not topics_path.exists():
        return jsonify({"status": "error", "message": "topics.yaml not found"}), 404

    with open(topics_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {"topics": []}

    initial_len = len(data.get("topics", []))
    data["topics"] = [t for t in data.get("topics", []) if t.get("name") != topic_name]

    if len(data["topics"]) == initial_len:
        return jsonify({"status": "error", "message": "Topic not found"}), 404

    with open(topics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    return jsonify({"status": "success", "message": f"Topic '{topic_name}' deleted successfully"})


@app.route("/api/config/domains", methods=["GET", "POST"])
def api_config_domains():
    """Get all domains or add/update a domain."""
    import yaml
    from flask import request

    domains_path = CONFIG_DIR / "domains.yaml"
    data = {"domains": {}}
    if domains_path.exists():
        with open(domains_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {"domains": {}}

    if request.method == "POST":
        domain_payload = request.get_json() or {}
        domain_name = domain_payload.get("domain_name", "").strip().lower()
        if not domain_name:
            return jsonify({"status": "error", "message": "Domain name is required"}), 400

        data["domains"][domain_name] = {
            "relevance_terms": domain_payload.get("relevance_terms", []),
            "gap_categories": domain_payload.get("gap_categories", ["methodology", "controls", "statistics", "reproducibility"]),
            "quality_metrics": domain_payload.get("quality_metrics", ["methodology_rigor", "statistical_power", "reproducibility_score"]),
        }

        with open(domains_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return jsonify({"status": "success", "message": f"Domain '{domain_name}' saved successfully", "domain": data["domains"][domain_name]})

    return jsonify(data.get("domains", {}))


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Trigger on-demand literature ingestion for a selected topic or custom query."""
    from flask import request
    from core.streaming_agent import fetch_europepmc_papers, is_domain_relevant, _is_within_time_window, _paper_key

    payload = request.get_json() or {}
    topic_name = payload.get("topic_name")
    custom_query = payload.get("query")
    time_window = int(payload.get("time_window_months", 240))
    limit = int(payload.get("limit", 20))

    if not topic_name and not custom_query:
        return jsonify({"status": "error", "message": "Either topic_name or query must be provided"}), 400

    topic_conf = get_topic_config_by_name(topic_name) if topic_name else {}
    domain = topic_conf.get("domain", "general")
    topic_terms = topic_conf.get("domain_terms", {})
    queries = [custom_query] if custom_query else (topic_conf.get("seed_queries") or [topic_name])

    memory = get_memory()
    ingested_papers = []
    total_candidates = 0

    for q in queries:
        candidates = fetch_europepmc_papers(q, limit=limit, time_window_months=time_window)
        total_candidates += len(candidates)
        for p in candidates:
            if not isinstance(p, dict):
                continue
            # Deduplication
            pkey = _paper_key(p)
            if memory.get_paper_by_key(pkey) or (p.get("doi") and memory.get_paper_by_key(p["doi"])):
                continue

            # Filtering
            if not _is_within_time_window(p, time_window):
                continue
            combined_text = f"{p.get('title', '')}\n{p.get('abstract', '')}"
            if not is_domain_relevant(combined_text, domain, topic_terms):
                continue

            record = memory.ingest_paper(p, topic_name=topic_name or "General Ingest", domain=domain, time_window_months=time_window)
            if record:
                ingested_papers.append({"title": p.get("title"), "doi": p.get("doi")})

    memory.save()
    return jsonify({
        "status": "success",
        "topic": topic_name,
        "total_candidates_fetched": total_candidates,
        "newly_ingested_count": len(ingested_papers),
        "papers": ingested_papers,
    })


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
