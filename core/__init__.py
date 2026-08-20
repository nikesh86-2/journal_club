"""
Journal Club Core Module
"""

from .literature_memory import JournalClubMemory
from .streaming_agent import (
    start_streaming,
    stop_streaming,
    stop_all_streaming,
    active_streams,
    start_all_topics,
)
from .paper_analyzer import (
    analyze_paper,
    analyze_batch,
    generate_summary,
    analyze_gaps,
    generate_critique,
    score_paper_quality,
)
from .recommendation_engine import (
    find_foundational_papers,
    detect_conflicting_papers,
    recommend_related_reading,
    cluster_papers_by_theme,
    generate_recommendations,
)
from .report_generator import (
    generate_topic_report,
    generate_paper_report,
    generate_summary_report,
    generate_all_reports,
)
from .training_data_collector import (
    TrainingDataCollector,
    collect_training_data_from_memory,
)
from .training_trigger import (
    check_and_trigger_training,
    run_training_pipeline,
)

__all__ = [
    "JournalClubMemory",
    "start_streaming",
    "stop_streaming",
    "stop_all_streaming",
    "active_streams",
    "start_all_topics",
    "analyze_paper",
    "analyze_batch",
    "generate_summary",
    "analyze_gaps",
    "generate_critique",
    "score_paper_quality",
    "find_foundational_papers",
    "detect_conflicting_papers",
    "recommend_related_reading",
    "cluster_papers_by_theme",
    "generate_recommendations",
    "generate_topic_report",
    "generate_paper_report",
    "generate_summary_report",
    "generate_all_reports",
    "TrainingDataCollector",
    "collect_training_data_from_memory",
    "check_and_trigger_training",
    "run_training_pipeline",
]
