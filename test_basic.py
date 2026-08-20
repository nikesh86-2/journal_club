#!/usr/bin/env python3
"""
Basic test for Journal Club pipeline components.
Tests core functionality without requiring external API calls.
"""

import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from core.literature_memory import JournalClubMemory
from core.paper_analyzer import analyze_paper, score_paper_quality, _rule_based_gap_analysis
from core.recommendation_engine import is_foundational_paper, find_foundational_papers


def test_literature_memory():
    """Test literature memory basic operations."""
    print("Testing Literature Memory...")

    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    memory = JournalClubMemory(temp_path)

    # Test paper ingestion
    from datetime import datetime
    current_year = str(datetime.now().year)
    test_paper = {
        "title": "Test Paper: RNA-Protein Interactions",
        "abstract": "This paper studies RNA-protein binding mechanisms using molecular dynamics simulations.",
        "year": current_year,
        "publication_date": f"{current_year}-06-01",
        "doi": "10.1234/test.2024.001",
        "source": "test",
    }

    record = memory.ingest_paper(
        test_paper,
        topic_name="RNA-Protein Interactions",
        domain="biophysics",
        time_window_months=12,
    )

    assert record is not None, "Paper ingestion failed"
    assert record["title"] == test_paper["title"], "Title mismatch"

    # Test retrieval
    papers = memory.get_papers_by_topic("RNA-Protein Interactions")
    assert len(papers) == 1, "Paper retrieval failed"

    # Test statistics
    stats = memory.get_statistics()
    assert stats["total_papers"] == 1, "Statistics incorrect"

    # Cleanup
    import os
    os.unlink(temp_path)

    print("✓ Literature Memory tests passed")
    return memory


def test_paper_analysis():
    """Test paper analysis components."""
    print("\nTesting Paper Analysis...")

    test_paper = {
        "title": "A Study of RNA Binding",
        "abstract": "This preliminary study investigates RNA-protein interactions using in vitro assays. No statistical analysis was performed.",
        "year": "2024",
    }

    # Test gap analysis (rule-based)
    gap_analysis = _rule_based_gap_analysis(
        f"{test_paper['title']} {test_paper['abstract']}"
    )

    assert isinstance(gap_analysis, dict), "Gap analysis should return dict"
    assert "methodology" in gap_analysis, "Missing methodology category"
    assert "statistics" in gap_analysis, "Missing statistics category"

    # Test quality scoring
    quality_scores = score_paper_quality(test_paper, gap_analysis)

    assert isinstance(quality_scores, dict), "Quality scores should return dict"
    assert "methodology_rigor" in quality_scores, "Missing methodology_rigor"
    assert "statistical_power" in quality_scores, "Missing statistical_power"

    print("✓ Paper Analysis tests passed")


def test_recommendation_engine():
    """Test recommendation engine components."""
    print("\nTesting Recommendation Engine...")

    # Test foundational paper detection
    old_paper = {
        "title": "Classic RNA Study",
        "year": "2015",
        "citation_count": 150,
    }

    recent_paper = {
        "title": "New RNA Study",
        "year": "2024",
        "citation_count": 5,
    }

    assert is_foundational_paper(old_paper) == True, "Old paper should be foundational"
    assert is_foundational_paper(recent_paper) == False, "Recent paper should not be foundational"

    # Test finding foundational papers
    papers = [old_paper, recent_paper]
    foundational = find_foundational_papers(papers)

    assert len(foundational) == 1, "Should find 1 foundational paper"
    assert foundational[0]["title"] == old_paper["title"], "Wrong foundational paper"

    print("✓ Recommendation Engine tests passed")


def test_configuration_loading():
    """Test configuration file loading."""
    print("\nTesting Configuration Loading...")

    import yaml

    # Test topics config
    topics_path = Path(__file__).parent / "config" / "topics.yaml"
    assert topics_path.exists(), "Topics config not found"

    with open(topics_path) as f:
        topics_data = yaml.safe_load(f)

    assert "topics" in topics_data, "Missing topics key"
    assert len(topics_data["topics"]) > 0, "No topics configured"

    # Test domains config
    domains_path = Path(__file__).parent / "config" / "domains.yaml"
    assert domains_path.exists(), "Domains config not found"

    with open(domains_path) as f:
        domains_data = yaml.safe_load(f)

    assert "domains" in domains_data, "Missing domains key"
    assert "biophysics" in domains_data["domains"], "Missing biophysics domain"

    print("✓ Configuration Loading tests passed")


def test_report_generation():
    """Test report generation components."""
    print("\nTesting Report Generation...")

    from core.report_generator import _format_paper_section

    test_paper = {
        "title": "Test Paper",
        "abstract": "Test abstract",
        "year": "2024",
        "summary": "This is a test summary.",
        "gap_analysis": {
            "methodology": ["Test gap"],
        },
        "quality_scores": {
            "methodology_rigor": 0.8,
        },
    }

    # Test paper section formatting
    lines = _format_paper_section(test_paper, 1)

    assert len(lines) > 0, "Paper section formatting failed"
    assert any("Test Paper" in line for line in lines), "Title not in formatted section"

    print("✓ Report Generation tests passed")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Journal Club Pipeline - Basic Tests")
    print("=" * 50)

    try:
        test_literature_memory()
        test_paper_analysis()
        test_recommendation_engine()
        test_configuration_loading()
        test_report_generation()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)

        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
