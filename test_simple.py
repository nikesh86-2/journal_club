#!/usr/bin/env python3
"""
Simple test for Journal Club pipeline - test components individually.
"""

import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from core.literature_memory import JournalClubMemory
    from core.paper_analyzer import analyze_paper, score_paper_quality
    from core.recommendation_engine import is_foundational_paper
    from core.report_generator import generate_topic_report
    print("✓ All modules imported successfully")


def test_configuration():
    """Test configuration files exist and are valid."""
    print("\nTesting configuration...")
    
    import yaml
    
    topics_path = Path(__file__).parent / "config" / "topics.yaml"
    assert topics_path.exists(), f"Topics config not found at {topics_path}"
    
    with open(topics_path) as f:
        data = yaml.safe_load(f)
    assert "topics" in data, "Missing topics key"
    print(f"✓ Topics config loaded with {len(data.get('topics', []))} topics")
    
    domains_path = Path(__file__).parent / "config" / "domains.yaml"
    assert domains_path.exists(), f"Domains config not found at {domains_path}"
    
    with open(domains_path) as f:
        data = yaml.safe_load(f)
    assert "domains" in data, "Missing domains key"
    print(f"✓ Domains config loaded with {len(data.get('domains', {}))} domains")


def test_gap_analysis():
    """Test gap analysis (rule-based, no LLM needed)."""
    print("\nTesting gap analysis...")
    
    from core.paper_analyzer import _rule_based_gap_analysis, score_paper_quality
    
    text = "This preliminary study uses in vitro assays. No statistical analysis was performed."
    
    gaps = _rule_based_gap_analysis(text)
    assert isinstance(gaps, dict) and len(gaps) > 0, "Gap analysis failed"
    print(f"✓ Gap analysis returned {len(gaps)} categories")
    
    scores = score_paper_quality({"title": "Test", "abstract": text}, gaps)
    assert isinstance(scores, dict) and len(scores) > 0, "Quality scoring failed"
    print(f"✓ Quality scores generated: {list(scores.keys())}")


def test_recommendation():
    """Test recommendation engine (rule-based)."""
    print("\nTesting recommendation engine...")
    
    from core.recommendation_engine import is_foundational_paper, find_foundational_papers
    
    old_paper = {"title": "Old", "year": "2015", "citation_count": 150}
    new_paper = {"title": "New", "year": "2024", "citation_count": 5}
    
    assert is_foundational_paper(old_paper) == True
    assert is_foundational_paper(new_paper) == False
    
    foundational = find_foundational_papers([old_paper, new_paper])
    assert len(foundational) == 1
    
    print("✓ Foundational paper detection works")


def main():
    """Run simple tests."""
    print("=" * 50)
    print("Journal Club - Simple Component Tests")
    print("=" * 50)
    
    test_imports()
    test_configuration()
    test_gap_analysis()
    test_recommendation()
    
    print("\n" + "=" * 50)
    print("✓ All simple tests passed!")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
