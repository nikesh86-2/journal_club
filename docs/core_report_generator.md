# Report Generator Module

**File**: `core/report_generator.py`

## Overview

The report generator creates markdown reports for topics, individual papers, and overall statistics. Reports are saved to the `output/markdown/` directory and can be downloaded via the web interface.

## Key Functions

### `generate_topic_report(topic_name, memory, output_dir)`

Generate a markdown report for a specific topic.

**Parameters:**
- `topic_name`: Topic name
- `memory`: JournalClubMemory instance
- `output_dir`: Output directory (default: `output/markdown`)

**Returns:**
- Path to generated report file

**Report Structure:**
```markdown
# Journal Club Report: {topic_name}

**Generated:** {timestamp}
**Total Papers:** {count}

## Statistics
- Papers analyzed: {count}/{total}
- Papers with gap analysis: {count}
- Papers with quality scores: {count}

## Papers
### {index}. {title}
**Authors:** {authors}
**Year:** {year}
**DOI:** [{doi}](https://doi.org/{doi})

#### Summary
{summary}

#### Gap Analysis
**Methodology:** {gap1}, {gap2}
**Statistics:** {gap1}

#### Quality Scores
- **Methodology Rigor:** {score}/1.0
- **Statistical Power:** {score}/1.0

#### Critique
{critique}
```

**Example:**
```python
from core.report_generator import generate_topic_report
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
report_path = generate_topic_report("RNA-Protein Interactions", memory)
print(f"Report saved to: {report_path}")
```

### `generate_paper_report(paper, output_dir)`

Generate a detailed markdown report for a single paper.

**Parameters:**
- `paper`: Paper dict with analysis results
- `output_dir`: Output directory (default: `output/markdown`)

**Returns:**
- Path to generated report file

**Report Structure:**
```markdown
# Paper Report: {title}

**Generated:** {timestamp}

## Metadata
**Title:** {title}
**Authors:** {authors}
**Year:** {year}
**DOI:** [{doi}](https://doi.org/{doi})
**PMID:** {pmid}
**URL:** {url}
**Publication Date:** {date}
**Topic:** {topic}
**Domain:** {domain}

## Abstract
{abstract}

## Summary
{summary}

## Gap Analysis
### Methodology
- {gap1}
- {gap2}

### Statistics
- {gap1}

## Quality Scores
- 🟢 **Methodology Rigor:** {score}/1.0
- 🟡 **Statistical Power:** {score}/1.0

## Critique
{critique}

## Recommendation Type
**{type}**

## Related Papers
- [{title}](https://doi.org/{doi})
```

**Example:**
```python
from core.report_generator import generate_paper_report

report_path = generate_paper_report(paper)
print(f"Report saved to: {report_path}")
```

### `generate_summary_report(memory, output_dir)`

Generate a summary statistics report.

**Parameters:**
- `memory`: JournalClubMemory instance
- `output_dir`: Output directory (default: `output/markdown`)

**Returns:**
- Path to generated report file

**Report Structure:**
```markdown
# Journal Club Summary Report

**Generated:** {timestamp}

## Overall Statistics
- **Total Papers:** {count}
- **Analyzed Papers:** {count}
- **Topics Tracked:** {count}
- **Domains Tracked:** {count}

## Papers by Topic
- **{topic}:** {count} papers
- **{topic}:** {count} papers

## Papers by Domain
- **{domain}:** {count} papers
- **{domain}:** {count} papers
```

**Example:**
```python
from core.report_generator import generate_summary_report
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
report_path = generate_summary_report(memory)
print(f"Report saved to: {report_path}")
```

### `generate_all_reports(memory, output_dir)`

Generate all reports (summary + topic reports).

**Parameters:**
- `memory`: JournalClubMemory instance
- `output_dir`: Output directory (default: `output/markdown`)

**Returns:**
- Dict mapping report type to file paths

**Behavior:**
1. Generates summary report
2. Generates topic report for each topic
3. Returns dict of generated file paths

**Example:**
```python
from core.report_generator import generate_all_reports
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
reports = generate_all_reports(memory)
print(f"Generated {len(reports)} reports")
```

## Helper Functions

### `_format_paper_section(paper, index)`

Format a paper section for topic reports.

**Parameters:**
- `paper`: Paper dict
- `index`: Paper number in list

**Returns:**
- List of markdown strings

**Behavior:**
- Formats metadata (authors, year, DOI)
- Formats summary if present
- Formats gap analysis with categories
- Formats quality scores with visual indicators
- Formats critique if present
- Formats recommendations if present
- Adds separator line

## Usage Patterns

### Generate Topic Report
```python
from core.report_generator import generate_topic_report
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
report_path = generate_topic_report("My Topic", memory)
```

### Generate Paper Report
```python
from core.report_generator import generate_paper_report

# Get paper from memory
paper = memory.get_paper_by_key("10.1234/example")
report_path = generate_paper_report(paper)
```

### Generate All Reports
```python
from core.report_generator import generate_all_reports
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
reports = generate_all_reports(memory)
for report_type, path in reports.items():
    print(f"{report_type}: {path}")
```

### Custom Output Directory
```python
from core.report_generator import generate_topic_report

report_path = generate_topic_report(
    "My Topic",
    memory,
    output_dir="/custom/output/path"
)
```

## Report Features

### Quality Score Indicators
- 🟢 High quality (≥ 0.8)
- 🟡 Medium quality (0.6 - 0.8)
- 🔴 Low quality (< 0.6)

### Gap Analysis Formatting
- Categories are bolded
- Gaps are bulleted
- Empty categories are omitted

### Metadata Formatting
- DOI is clickable link
- URL is clickable link
- PMID is plain text
- Publication date is formatted

### Critique Formatting
- Full critique in dedicated section
- Includes gap analysis context if available

## Error Handling

- Missing fields in paper dict are skipped
- Empty sections are omitted
- File I/O errors are caught and logged
- Invalid paper keys are handled gracefully

## Output Structure

```
output/markdown/
├── summary_report.md
├── {topic_name}_report.md
├── paper_{doi}.md
└── paper_{safe_title}.md
```

## File Naming

### Topic Reports
- Format: `{topic_name}_report.md`
- Spaces replaced with underscores
- Lowercase

### Paper Reports
- If DOI present: `paper_{doi}.md`
- If DOI absent: `paper_{safe_title}.md`
- DOI slashes replaced with underscores
- Title limited to 50 characters
- Non-alphanumeric characters removed

## Performance Considerations

- Report generation is I/O-bound
- Large topic reports may take time to write
- Consider generating reports asynchronously for large datasets
- Memory usage scales with number of papers

## Customization

### Modify Report Template
Edit `_format_paper_section` to change topic report format.

### Modify Paper Report Template
Edit `generate_paper_report` to change paper report format.

### Add Custom Sections
Add new sections to report functions and update formatting helpers.

## Troubleshooting

### Report file not created
- Check output directory permissions
- Verify memory has papers for the topic
- Check disk space

### Report is empty
- Verify paper has analysis results
- Check that summary/gap analysis exist
- Verify quality scores are present

### File naming issues
- Check for special characters in titles
- Verify DOI format
- Check for very long titles
