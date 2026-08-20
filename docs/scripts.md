# Scripts Documentation

## Overview

The scripts directory contains execution scripts for setting up the environment and running the Journal Club pipeline.

## Scripts

### `setup.sh`

Environment setup script that creates directories, copies configuration files, and installs dependencies.

#### Usage

```bash
./scripts/setup.sh
```

#### What It Does

1. **Create Directories**
   - `training/journal_club_data/`
   - `training/journal_club_hf_dataset/`
   - `training/journal_club_output/`
   - `training/journal_club_merged_model/`
   - `output/markdown/`
   - `output/json/`
   - `cache/`

2. **Copy Environment File**
   - Copies `.env.example` to `.env` if `.env` doesn't exist
   - Preserves existing `.env` if it exists

3. **Install Dependencies**
   - Installs packages from `requirements.txt`
   - Uses pip with `-r` flag

4. **Make Scripts Executable**
   - Makes `setup.sh` executable
   - Makes `run_journal_club.sh` executable

#### Requirements

- Bash shell
- Python 3.8+
- pip
- Internet connection (for pip install)

#### Example Output

```
Creating directories...
✓ Created training/journal_club_data
✓ Created training/journal_club_hf_dataset
✓ Created training/journal_club_output
✓ Created training/journal_club_merged_model
✓ Created output/markdown
✓ Created output/json
✓ Created cache

Setting up environment file...
✓ .env file ready

Installing dependencies...
Collecting langchain-community...
...
Successfully installed ...

Making scripts executable...
✓ Scripts are executable

Setup complete!
```

#### Troubleshooting

**Permission Denied Error**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Pip Install Fails**
- Check Python version: `python3 --version`
- Upgrade pip: `pip install --upgrade pip`
- Install manually: `pip install -r requirements.txt`

**Directory Creation Fails**
- Check write permissions in current directory
- Manually create directories

---

### `run_journal_club.sh`

Main execution script that orchestrates the Journal Club pipeline components.

#### Usage

```bash
./scripts/run_journal_club.sh [command]
```

#### Commands

| Command | Description |
|---------|-------------|
| `streaming` | Start literature streaming only |
| `analysis` | Run paper analysis only |
| `reports` | Generate markdown reports only |
| `web` | Start web interface only |
| `training` | Check threshold and trigger LoRA training |
| `collect-data` | Collect training data from literature memory |
| `convert-dataset` | Convert training data to HuggingFace format |
| `all` | Run full pipeline (default) |
| `all-with-training` | Run full pipeline + training if threshold met |

#### Default Behavior

If no command is specified, runs `all`:
1. Start literature streaming
2. Run paper analysis
3. Generate reports
4. Start web interface

#### Environment Setup

The script automatically:
- Loads environment variables from `.env`
- Sets defaults for missing variables
- Adds VLAB2 to Python path if it exists
- Checks and installs dependencies

#### Functions

##### `start_streaming()`

Starts literature streaming for all configured topics.

**Behavior:**
- Loads topics from `config/topics.yaml`
- Starts streaming thread for each topic
- Logs number of streams started

**Example:**
```bash
./scripts/run_journal_club.sh streaming
```

##### `run_analysis()`

Runs paper analysis on unanalyzed papers.

**Behavior:**
- Loads literature memory
- Gets statistics
- Iterates through topics
- Analyzes papers without summaries
- Updates memory with analysis results

**Example:**
```bash
./scripts/run_journal_club.sh analysis
```

##### `generate_reports()`

Generates markdown reports for all topics and summary.

**Behavior:**
- Loads literature memory
- Calls `generate_all_reports()`
- Logs number of reports generated

**Example:**
```bash
./scripts/run_journal_club.sh reports
```

##### `start_web()`

Starts the Flask web server.

**Behavior:**
- Starts Flask app on configured port
- Blocks until server stopped

**Example:**
```bash
./scripts/run_journal_club.sh web
```

##### `trigger_training()`

Checks training threshold and triggers training if met.

**Behavior:**
- Calls `check_and_trigger_training()`
- Logs training status

**Example:**
```bash
./scripts/run_journal_club.sh training
```

##### `collect_training_data()`

Collects training data from literature memory.

**Behavior:**
- Calls `collect_training_data_from_memory()`
- Logs number of examples collected

**Example:**
```bash
./scripts/run_journal_club.sh collect-data
```

##### `convert_dataset()`

Converts training data to HuggingFace format.

**Behavior:**
- Runs `convert_journal_club_dataset.py`
- Logs conversion status

**Example:**
```bash
./scripts/run_journal_club.sh convert-dataset
```

#### Examples

##### Run Full Pipeline
```bash
./scripts/run_journal_club.sh all
```

##### Run with Training
```bash
./scripts/run_journal_club.sh all-with-training
```

##### Individual Components
```bash
# Just streaming
./scripts/run_journal_club.sh streaming

# Just analysis
./scripts/run_journal_club.sh analysis

# Just web interface
./scripts/run_journal_club.sh web
```

##### Manual Training Pipeline
```bash
# Step 1: Collect data
./scripts/run_journal_club.sh collect-data

# Step 2: Convert dataset
./scripts/run_journal_club.sh convert-dataset

# Step 3: Trigger training
./scripts/run_journal_club.sh training
```

#### Environment Variables

The script respects these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_FAISS_INDEX_PATH` | `/scratch/fbsnpat/bot/VLAB2/cache/faiss_index` | FAISS index path |
| `JOURNAL_CLUB_TIME_WINDOW_MONTHS` | `12` | Time window in months |
| `JOURNAL_CLUB_WEB_PORT` | `5000` | Web server port |
| `JOURNAL_CLUB_LITERATURE_MEMORY_PATH` | `literature_memory.json` | Memory file path |

#### Error Handling

- Missing dependencies: Attempts to install
- Missing VLAB2: Continues without VLAB2 integration
- Python errors: Logs and exits with error code
- Missing .env: Uses defaults

#### Troubleshooting

**Command Not Found**
```bash
chmod +x scripts/run_journal_club.sh
./scripts/run_journal_club.sh all
```

**Python Module Not Found**
- Ensure `PYTHONPATH` includes current directory
- Run `./scripts/setup.sh` to install dependencies
- Check Python version

**Streaming Not Starting**
- Check Semantic Scholar API key in `.env`
- Verify topics are configured in `config/topics.yaml`
- Check network connectivity

**Analysis Fails**
- Ensure papers have been ingested
- Check LLM API key
- Verify memory file exists

**Web Server Won't Start**
- Check if port is already in use
- Verify Flask is installed
- Check web app configuration

**Training Not Triggering**
- Check `JOURNAL_CLUB_LORA_TRAIN=1` in `.env`
- Verify paper count meets threshold
- Check quality score threshold

---

## Script Development

### Adding New Commands

To add a new command to `run_journal_club.sh`:

1. **Add function:**
```bash
my_function() {
    echo "Running my function..."
    python3 -c "import my_module; my_module.my_function()"
}
```

2. **Add to case statement:**
```bash
my_command)
    my_function
    ;;
```

3. **Update usage message:**
```bash
echo "  my_command  - Run my custom function"
```

### Adding Dependencies

To add new dependencies:

1. **Add to `requirements.txt`:**
```
new-package==1.0.0
```

2. **Re-run setup:**
```bash
./scripts/setup.sh
```

### Customizing Setup

To customize `setup.sh`:

1. **Add directory creation:**
```bash
mkdir -p my_new_directory
```

2. **Add file copying:**
```bash
if [ ! -f "my_config.yaml" ]; then
    cp "my_config.example.yaml" "my_config.yaml"
fi
```

3. **Add custom installation:**
```bash
pip install my-custom-package
```

## Best Practices

### 1. Idempotency
Scripts should be safe to run multiple times:
- Check if directories exist before creating
- Check if files exist before copying
- Use `|| true` for non-critical commands

### 2. Error Handling
Check command exit codes:
```bash
if ! command; then
    echo "Error: command failed"
    exit 1
fi
```

### 3. Logging
Log important steps:
```bash
echo "✓ Step completed"
echo "✗ Step failed"
```

### 4. Configuration
Use environment variables for configuration:
```bash
export MY_VAR="${MY_VAR:-default_value}"
```

### 5. Dependencies
Check for required tools:
```bash
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi
```

## Security Considerations

- **API Keys**: Never commit `.env` file with real keys
- **Permissions**: Scripts should be executable but not world-writable
- **Input Validation**: Validate user input in scripts
- **Path Traversal**: Use absolute paths where possible

## Performance Considerations

- **Parallel Execution**: Consider running independent steps in parallel
- **Caching**: Cache expensive operations
- **Batch Processing**: Process items in batches for efficiency
- **Resource Limits**: Monitor memory and CPU usage

## Monitoring

### Logging

Scripts log to stdout/stderr. To capture logs:
```bash
./scripts/run_journal_club.sh all 2>&1 | tee pipeline.log
```

### Process Monitoring

Check running processes:
```bash
ps aux | grep journal_club
```

### Resource Monitoring

Monitor resource usage:
```bash
top -p $(pgrep -f journal_club)
```
