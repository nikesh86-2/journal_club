# Journal Club Pipeline Documentation

This directory contains comprehensive documentation for the Journal Club pipeline.

## Documentation Index

### Architecture & Overview
- **[overview.md](overview.md)** - System architecture, data flow, component hierarchy, design principles, technology stack, and integration with VLAB2

### Core Modules
- **[core_literature_memory.md](core_literature_memory.md)** - Persistent storage for papers, analysis results, and recommendations
- **[core_streaming_agent.md](core_streaming_agent.md)** - Literature ingestion with domain filtering and time-based filtering
- **[core_paper_analyzer.md](core_paper_analyzer.md)** - Gap analysis, quality scoring, and critique generation
- **[core_recommendation_engine.md](core_recommendation_engine.md)** - Foundational/conflicting paper detection and related reading recommendations
- **[core_report_generator.md](core_report_generator.md)** - Markdown report generation for topics and papers
- **[core_training_data_collector.md](core_training_data_collector.md)** - Training data collection for LoRA fine-tuning
- **[core_training_trigger.md](core_training_trigger.md)** - Training orchestration and threshold monitoring

### Web Interface
- **[web_app.md](web_app.md)** - Flask web application with dashboard, topic views, and API endpoints

### Configuration
- **[config.md](config.md)** - YAML configuration files (topics, domains, settings) and environment variables

### Scripts
- **[scripts.md](scripts.md)** - Setup and execution scripts for the pipeline

### Optimization
- **[optimizations.md](optimizations.md)** - Performance optimization recommendations, critical issues, and implementation priorities

## Quick Start

1. **Read the [overview.md](overview.md)** to understand the system architecture
2. **Configure the pipeline** using [config.md](config.md) as a guide
3. **Run setup** using instructions in [scripts.md](scripts.md)
4. **Start the pipeline** using `./scripts/run_journal_club.sh all`

## Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                      Configuration Layer                     │
│                    (config.md)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Core Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Literature   │  │ Streaming    │  │ Paper        │     │
│  │ Memory       │  │ Agent        │  │ Analyzer     │     │
│  │ (literature  │  │ (streaming   │  │ (paper_      │     │
│  │  _memory.md) │  │  _agent.md)  │  │  analyzer.md)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Recommend-   │  │ Report       │  │ Training     │     │
│  │ ation       │  │ Generator    │  │ Data         │     │
│  │ (recommend-  │  │ (report_     │  │ Collector    │     │
│  │  _engine.md) │  │  generator.md)│  │ (training_   │     │
│  └──────────────┘  └──────────────┘  │  data_       │     │
│  ┌──────────────┐                      │  collector.md)│     │
│  │ Training     │                      └──────────────┘     │
│  │ Trigger      │                      ┌──────────────┐     │
│  │ (training_   │                      │ Training     │     │
│  │  _trigger.md)│                      │ Trigger      │     │
│  └──────────────┘                      │ (training_   │     │
│                                         │  _trigger.md)│     │
│                                         └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                       │
│                    (web_app.md)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Execution Layer                          │
│                      (scripts.md)                             │
└─────────────────────────────────────────────────────────────┘
```

## Documentation Conventions

### Code Blocks
- **Python code**: Uses syntax highlighting for Python
- **Bash commands**: Uses syntax highlighting for bash
- **YAML**: Uses syntax highlighting for YAML
- **JSON**: Uses syntax highlighting for JSON

### Parameter Tables
Parameters are documented in tables with:
- **Field**: Parameter name
- **Type**: Data type
- **Required**: Whether the parameter is required
- **Description**: What the parameter does

### Example Sections
Each module includes:
- Basic usage examples
- Advanced usage patterns
- Common use cases
- Integration examples

### Error Handling
Each module documents:
- Common errors
- Error handling strategies
- Troubleshooting steps

## Contributing to Documentation

When adding new features or modifying existing ones:

1. Update the relevant module documentation
2. Add examples for new functionality
3. Update the architecture overview if components change
4. Update configuration documentation if new settings are added
5. Update script documentation if new commands are added

## Additional Resources

- **Main README**: `../README.md` - Project overview and quick start
- **Requirements**: `../requirements.txt` - Python dependencies
- **Environment Example**: `../.env.example` - Environment variable template
- **Training Config**: `../training/journal_club_training_config.yaml` - LoRA training configuration

## Support

For issues or questions:
1. Check the relevant module documentation
2. Review the troubleshooting sections
3. Check the configuration documentation
4. Review the script documentation
