"""
training_data_collector.py

Collects training data from analyzed papers for LoRA fine-tuning.

Converts analyzed papers into instruction-tuning format:
- Summarization tasks
- Gap analysis tasks
- Critique generation tasks
- Quality scoring tasks
- Recommendation tasks
- QA pairs
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("journal_club.training")


class TrainingDataCollector:
    """Collects training data from analyzed papers for LoRA fine-tuning."""
    
    def __init__(self, output_dir: str | None = None):
        if output_dir is None:
            output_dir = Path(__file__).parents[1] / "training" / "journal_club_data"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def _write_example(self, example: Dict[str, Any], filename: str) -> None:
        """Write a single training example to JSONL file."""
        output_path = self.output_dir / filename
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    def collect_summarization(
        self,
        paper: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Create a summarization training example."""
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        summary = paper.get("summary")
        
        if not title or not abstract or not summary:
            return None
        
        example = {
            "instruction": "Summarize the following research paper in 2-3 sentences, focusing on the main research question, methods, and key findings.",
            "input": f"Title: {title}\n\nAbstract: {abstract}",
            "output": summary,
            "metadata": {
                "type": "summarization",
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": paper.get("topic_name"),
                "domain": paper.get("domain"),
                "paper_doi": paper.get("doi"),
                "quality_score": paper.get("quality_scores", {}).get("overall_quality"),
            }
        }
        
        filename = f"summarization_{self.session_id}.jsonl"
        self._write_example(example, filename)
        
        return example
    
    def collect_gap_analysis(
        self,
        paper: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Create a gap analysis training example."""
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        gap_analysis = paper.get("gap_analysis")
        
        if not title or not abstract or not gap_analysis:
            return None
        
        # Convert gap analysis to JSON string
        gap_json = json.dumps(gap_analysis, indent=2)
        
        example = {
            "instruction": "Analyze the following research paper for gaps in methodology, controls, statistics, and reproducibility. Identify specific issues in each category.",
            "input": f"Title: {title}\n\nAbstract: {abstract}",
            "output": gap_json,
            "metadata": {
                "type": "gap_analysis",
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": paper.get("topic_name"),
                "domain": paper.get("domain"),
                "paper_doi": paper.get("doi"),
            }
        }
        
        filename = f"gap_analysis_{self.session_id}.jsonl"
        self._write_example(example, filename)
        
        return example
    
    def collect_critique(
        self,
        paper: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Create a critique generation training example."""
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        critique = paper.get("critique")
        gap_analysis = paper.get("gap_analysis", {})
        
        if not title or not abstract or not critique:
            return None
        
        # Include gap analysis in input for context
        gap_text = ""
        for category, issues in gap_analysis.items():
            if issues:
                gap_text += f"\n{category.capitalize()}: {', '.join(issues)}"
        
        input_text = f"Title: {title}\n\nAbstract: {abstract}"
        if gap_text:
            input_text += f"\n\nIdentified gaps:{gap_text}"
        
        example = {
            "instruction": "Generate a constructive critique of the research paper, highlighting strengths, discussing identified gaps, and suggesting improvements.",
            "input": input_text,
            "output": critique,
            "metadata": {
                "type": "critique",
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": paper.get("topic_name"),
                "domain": paper.get("domain"),
                "paper_doi": paper.get("doi"),
            }
        }
        
        filename = f"critique_{self.session_id}.jsonl"
        self._write_example(example, filename)
        
        return example
    
    def collect_quality_scoring(
        self,
        paper: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Create a quality scoring training example."""
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        quality_scores = paper.get("quality_scores")
        gap_analysis = paper.get("gap_analysis", {})
        
        if not title or not abstract or not quality_scores:
            return None
        
        # Convert quality scores to JSON string
        scores_json = json.dumps(quality_scores, indent=2)
        
        # Include gap analysis in input for context
        gap_text = ""
        for category, issues in gap_analysis.items():
            if issues:
                gap_text += f"\n{category.capitalize()}: {', '.join(issues)}"
        
        input_text = f"Title: {title}\n\nAbstract: {abstract}"
        if gap_text:
            input_text += f"\n\nIdentified gaps:{gap_text}"
        
        example = {
            "instruction": "Score the paper on methodology rigor, statistical power, reproducibility, and control quality. Provide scores from 0 to 1 for each metric.",
            "input": input_text,
            "output": scores_json,
            "metadata": {
                "type": "quality_scoring",
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": paper.get("topic_name"),
                "domain": paper.get("domain"),
                "paper_doi": paper.get("doi"),
            }
        }
        
        filename = f"quality_scoring_{self.session_id}.jsonl"
        self._write_example(example, filename)
        
        return example
    
    def collect_recommendations(
        self,
        paper: Dict[str, Any],
        recommendations: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any] | None:
        """Create a recommendation training example."""
        
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        if not title or not abstract:
            return None
        
        # Format recommendations as output
        rec_output = {
            "foundational": [
                {"title": p.get("title"), "doi": p.get("doi")}
                for p in recommendations.get("foundational", [])[:3]
            ],
            "conflicting": [
                {"title": p.get("title"), "doi": p.get("doi")}
                for p in recommendations.get("conflicting", [])[:3]
            ],
            "related": [
                {"title": p.get("title"), "doi": p.get("doi")}
                for p in recommendations.get("related", [])[:5]
            ],
        }
        
        rec_json = json.dumps(rec_output, indent=2)
        
        example = {
            "instruction": "Recommend foundational papers, conflicting papers, and related reading for the given research paper.",
            "input": f"Title: {title}\n\nAbstract: {abstract}",
            "output": rec_json,
            "metadata": {
                "type": "recommendations",
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": paper.get("topic_name"),
                "domain": paper.get("domain"),
                "paper_doi": paper.get("doi"),
            }
        }
        
        filename = f"recommendations_{self.session_id}.jsonl"
        self._write_example(example, filename)
        
        return example
    
    def collect_qa_pair(
        self,
        paper: Dict[str, Any],
        question: str,
        answer: str,
    ) -> Dict[str, Any] | None:
        """Create a QA pair training example."""
        
        if not question or not answer:
            return None
        
        # Include paper context in input
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        input_text = f"Title: {title}\n\nAbstract: {abstract}\n\nQuestion: {question}"
        
        example = {
            "instruction": "Answer the question based on the research paper.",
            "input": input_text,
            "output": answer,
            "metadata": {
                "type": "qa_pair",
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": paper.get("topic_name"),
                "domain": paper.get("domain"),
                "paper_doi": paper.get("doi"),
            }
        }
        
        filename = f"qa_pair_{self.session_id}.jsonl"
        self._write_example(example, filename)
        
        return example
    
    def collect_all(
        self,
        paper: Dict[str, Any],
        recommendations: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Collect all training examples for a paper."""
        
        examples = []
        
        # Collect each type of example
        if paper.get("summary"):
            ex = self.collect_summarization(paper)
            if ex:
                examples.append(ex)
        
        if paper.get("gap_analysis"):
            ex = self.collect_gap_analysis(paper)
            if ex:
                examples.append(ex)
        
        if paper.get("critique"):
            ex = self.collect_critique(paper)
            if ex:
                examples.append(ex)
        
        if paper.get("quality_scores"):
            ex = self.collect_quality_scoring(paper)
            if ex:
                examples.append(ex)
        
        if recommendations:
            ex = self.collect_recommendations(paper, recommendations)
            if ex:
                examples.append(ex)
        
        log.info("Collected %d training examples for paper: %s", len(examples), paper.get("title", "")[:50])
        
        return examples
    
    def collect_batch(
        self,
        papers: List[Dict[str, Any]],
        recommendations_map: Dict[str, Dict[str, List[Dict[str, Any]]]] | None = None,
    ) -> int:
        """Collect training examples for a batch of papers."""
        
        total_examples = 0
        
        for paper in papers:
            paper_key = paper.get("doi") or paper.get("title")
            recommendations = recommendations_map.get(paper_key) if recommendations_map else None
            
            examples = self.collect_all(paper, recommendations)
            total_examples += len(examples)
        
        log.info("Collected %d total training examples from %d papers", total_examples, len(papers))
        
        return total_examples


def collect_training_data_from_memory(
    memory_path: str | None = None,
    output_dir: str | None = None,
    min_quality_score: float = 0.5,
) -> int:
    """Collect training data from literature memory."""
    
    from .literature_memory import JournalClubMemory
    
    memory = JournalClubMemory(memory_path)
    stats = memory.get_statistics()
    
    collector = TrainingDataCollector(output_dir)
    
    total_examples = 0
    
    # Collect papers by topic
    for topic in stats["by_topic"].keys():
        papers = memory.filter_papers(topic=topic, limit=1000)
        
        # Filter by quality score if available
        high_quality_papers = [
            p for p in papers
            if p.get("quality_scores", {}).get("overall_quality", 0) >= min_quality_score
        ]
        
        if not high_quality_papers:
            continue
        
        examples = collector.collect_batch(high_quality_papers)
        total_examples += examples
    
    return total_examples
