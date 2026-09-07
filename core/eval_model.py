"""
eval_model.py

Compare a base model against a fine-tuned (merged) model on held-out papers.

Metrics computed per model:
  - summary non-empty rate (fraction of summaries with real content)
  - gap analysis JSON parse rate (structured-output reliability)
  - average summary length
  - optional LLM-judge scores (summary fidelity and gap plausibility, 1-5)

Benchmark papers are taken from the literature memory, excluding DOIs that
appear in the locked training split.

Usage:
    python core/eval_model.py --base /path/to/base --merged /path/to/merged
    python core/eval_model.py --base A --merged B --papers 10 --judge-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("journal_club.eval")

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

SUMMARY_SYSTEM = "You are an expert scientific summarizer. Generate clear, concise summaries of research papers."
GAPS_SYSTEM = "You are an expert critical reviewer. Identify methodological and statistical gaps in research papers."


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------

def load_benchmark_papers(n: int = 10) -> list:
    """Load held-out papers (title + abstract) from memory, excluding any
    DOI that appears in the locked training split."""
    from core.literature_memory import JournalClubMemory

    memory = JournalClubMemory()
    papers = memory.get_all_papers()

    train_dois = set()
    try:
        split_path = _REPO_ROOT / "training" / "journal_club_data" / "locked_split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_dois = {str(d).strip().lower() for d in split.get("train_dois", [])}
    except Exception:
        log.warning("Could not load locked split; using all papers as candidates")

    candidates = []
    for p in papers:
        doi = str(p.get("doi") or "").strip().lower()
        if doi in train_dois:
            continue
        if p.get("title") and p.get("abstract"):
            candidates.append(
                {"title": p["title"], "abstract": p["abstract"], "doi": p.get("doi")}
            )
    return candidates[:n]


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def _load_pipeline(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    device_map = "auto" if cuda else None

    log.info("Loading model for evaluation: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.3,
        do_sample=False,
    )


def _build_summary_messages(paper: dict) -> list:
    prompt = (
        "Generate a concise 2-3 sentence summary of the following paper:\n\n"
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        "Focus on:\n"
        "- Main research question/objective\n"
        "- Key methods used\n"
        "- Primary findings/conclusions\n"
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def _build_gaps_messages(paper: dict) -> list:
    prompt = (
        "Analyze the following research paper for potential gaps and weaknesses:\n\n"
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        "Identify specific issues in these categories:\n\n"
        "1. Methodology gaps\n2. Missing controls\n3. Statistical issues\n"
        "4. Reproducibility concerns\n\n"
        "Format your response as a JSON object with keys: methodology, controls, "
        "statistics, reproducibility. Each key should have a list of strings.\n"
    )
    return [
        {"role": "system", "content": GAPS_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def _generate(pipeline_obj, messages: list) -> str:
    """Generate text with reasoning mode disabled (where supported)."""
    tokenizer = pipeline_obj.tokenizer
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    outputs = pipeline_obj(
        prompt,
        max_new_tokens=1024,
        do_sample=False,
        return_full_text=False,
    )
    text = ""
    if outputs and isinstance(outputs[0], dict):
        text = outputs[0].get("generated_text", "")
    return _THINK_RE.sub("", text or "").strip()


def _extract_json(text: str):
    text = _THINK_RE.sub("", text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find the first {...} span
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# LLM-as-judge (optional)
# ---------------------------------------------------------------------------

def _judge(paper: dict, summary: str, gaps: dict, judge_url: str):
    """Score the generated outputs (1-5 each) using the external judge LLM."""
    import requests

    gap_text = json.dumps(gaps) if gaps else "{}"
    prompt = (
        f"Paper title: {paper['title']}\n\n"
        f"Paper abstract: {paper['abstract'][:1500]}\n\n"
        f"Generated summary: {summary[:1000]}\n\n"
        f"Generated gap analysis (JSON): {gap_text[:1000]}\n\n"
        "Rate ONLY the quality of the generated outputs on two axes, each 1-5 "
        "(5 = excellent): summary_score (accurate, covers question/methods/findings), "
        'gap_score (plausible, specific, correctly categorized). Respond with JSON: '
        '{"summary_score": int, "gap_score": int}'
    )
    try:
        resp = requests.post(
            f"{judge_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator of scientific text quality."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_json(content) or {}
        return int(parsed.get("summary_score", 0)), int(parsed.get("gap_score", 0))
    except Exception as e:
        log.warning("Judge call failed: %s", e)
        return 0, 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_papers(pipeline_obj, papers: list, judge_url: str | None = None) -> list:
    rows = []
    for i, paper in enumerate(papers, 1):
        log.info("Evaluating paper %d/%d: %s", i, len(papers), paper["title"][:60])
        summary = _generate(pipeline_obj, _build_summary_messages(paper))
        gaps_text = _generate(pipeline_obj, _build_gaps_messages(paper))
        gaps = _extract_json(gaps_text)
        gaps_ok = (
            isinstance(gaps, dict)
            and all(
                isinstance(gaps.get(k), list)
                for k in ("methodology", "controls", "statistics", "reproducibility")
            )
        )
        gaps_dict: dict = gaps if (gaps_ok and isinstance(gaps, dict)) else {}
        row = {
            "doi": paper.get("doi"),
            "summary_ok": bool(summary) and len(summary) >= 20,
            "summary_len": len(summary or ""),
            "gaps_ok": gaps_ok,
            "summary": (summary or "")[:500],
            "gaps": gaps_dict,
        }
        if judge_url:
            row["judge_summary"], row["judge_gaps"] = _judge(
                paper, summary or "", gaps_dict, judge_url
            )
        rows.append(row)
    return rows


def _aggregate(rows: list, name: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"model": name, "papers": 0}

    judge_summary = sum(r.get("judge_summary", 0) for r in rows) / n
    judge_gaps = sum(r.get("judge_gaps", 0) for r in rows) / n
    return {
        "model": name,
        "papers": n,
        "summary_nonempty_rate": round(sum(1 for r in rows if r["summary_ok"]) / n, 3),
        "gap_parse_rate": round(sum(1 for r in rows if r["gaps_ok"]) / n, 3),
        "avg_summary_chars": round(sum(r["summary_len"] for r in rows) / n, 1),
        "judge_summary_mean": round(judge_summary, 2),
        "judge_gap_mean": round(judge_gaps, 2),
    }


def _release_model(pipeline_obj) -> None:
    try:
        del pipeline_obj
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_models(
    base_path: str,
    merged_path: str,
    papers: int = 10,
    judge_url: str | None = None,
) -> dict:
    """Evaluate base vs merged model on held-out papers.

    Returns a report dict including a 'merged_better' bool used by the merge
    step to decide whether to activate the new model.
    """
    benchmark = load_benchmark_papers(papers)
    report: dict = {"benchmark_papers": len(benchmark)}
    if not benchmark:
        log.warning("No benchmark papers available; skipping evaluation")
        report["merged_better"] = True
        return report

    for path, name in ((base_path, "base"), (merged_path, "merged")):
        log.info("Evaluating %s model: %s", name, path)
        pipe = None
        try:
            pipe = _load_pipeline(path)
            rows = _evaluate_papers(pipe, benchmark, judge_url=judge_url)
            report[name] = _aggregate(rows, name)
        except Exception as e:
            log.error("Evaluation of %s failed: %s", name, e)
            report[name] = {"model": name, "error": str(e)}
        finally:
            if pipe is not None:
                _release_model(pipe)

    base = report.get("base", {})
    merged = report.get("merged", {})

    if merged.get("error"):
        better = False
    elif base.get("error"):
        better = True
    elif not merged.get("papers") or not base.get("papers"):
        better = bool(merged.get("papers"))
    else:
        # Must not regress on structured-output reliability...
        better = (
            merged["gap_parse_rate"] >= base["gap_parse_rate"]
            and merged["summary_nonempty_rate"] >= base["summary_nonempty_rate"]
        )
        # ...and, when judge scores are available, no meaningful quality drop.
        if base.get("judge_summary_mean") and merged.get("judge_summary_mean"):
            better = better and (
                merged["judge_summary_mean"] >= base["judge_summary_mean"] - 0.5
            )

    report["merged_better"] = better
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import os

    parser = argparse.ArgumentParser(
        description="Compare a base model against a merged/fine-tuned model on held-out papers."
    )
    parser.add_argument("--base", required=True, help="Path to the base model")
    parser.add_argument("--merged", required=True, help="Path to the merged model")
    parser.add_argument("--papers", type=int, default=10, help="Number of held-out papers to use")
    parser.add_argument(
        "--judge-url",
        default=None,
        help="llama-server URL to use as an LLM judge (default: JOURNAL_CLUB_LLAMA_SERVER_URL)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    judge_url = args.judge_url or os.getenv("JOURNAL_CLUB_LLAMA_SERVER_URL")
    report = evaluate_models(
        args.base,
        args.merged,
        papers=args.papers,
        judge_url=judge_url,
    )
    print(json.dumps(report, indent=2))

    if report.get("merged_better"):
        print("\nResult: merged model passed the evaluation bar")
        return 0
    print("\nResult: merged model did NOT meet the evaluation bar")
    return 1


if __name__ == "__main__":
    sys.exit(main())
