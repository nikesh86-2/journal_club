# Training Trigger Module

**File**: `core/training_trigger.py`

## Overview

The training trigger monitors the literature memory and automatically triggers LoRA fine-tuning when the paper count threshold is reached. It orchestrates the complete training pipeline: data collection, dataset conversion, LoRA training, and model merging.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_LORA_TRAIN` | `0` | Enable automatic LoRA training (1 = yes) |
| `JOURNAL_CLUB_MIN_TRAIN_PAPERS` | `200` | Minimum papers required for training |
| `JOURNAL_CLUB_MIN_QUALITY_SCORE` | `0.5` | Minimum quality score for training data |
| `JOURNAL_CLUB_USE_FINETUNED` | `0` | Use fine-tuned model after training |
| `JOURNAL_CLUB_FINETUNED_MODEL_PATH` | `training/journal_club_merged_model` | Path to merged model |
| `JOURNAL_CLUB_FALLBACK_TO_BASE` | `1` | Fall back to base model if fine-tuned fails |

### Paths

- `VLAB2_PATH`: Path to VLAB2 directory (default: `../VLAB2`)
- `TRAINING_DATA_DIR`: Training data directory (default: `training/journal_club_data`)
- `HF_DATASET_PATH`: HuggingFace dataset directory (default: `training/journal_club_hf_dataset`)
- `OUTPUT_MODEL_PATH`: LoRA adapter output (default: `training/journal_club_output`)
- `MERGED_MODEL_PATH`: Merged model output (default: `training/journal_club_merged_model`)

## Key Functions

### `check_training_threshold(memory)`

Check if training threshold is met.

**Parameters:**
- `memory`: JournalClubMemory instance

**Returns:**
- `True` if threshold met and training enabled, `False` otherwise

**Behavior:**
- Checks if `JOURNAL_CLUB_LORA_TRAIN=1`
- Checks if total papers ≥ `MIN_TRAIN_PAPERS`
- Logs status

**Example:**
```python
from core.training_trigger import check_training_threshold
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
if check_training_threshold(memory):
    print("Training threshold met")
```

### `collect_training_data(memory)`

Collect training data from literature memory.

**Parameters:**
- `memory`: JournalClubMemory instance

**Returns:**
- Number of training examples collected

**Behavior:**
- Calls `collect_training_data_from_memory`
- Filters by quality score
- Saves to `TRAINING_DATA_DIR`
- Logs count

**Example:**
```python
count = collect_training_data(memory)
print(f"Collected {count} examples")
```

### `convert_to_hf_dataset()`

Convert JSONL data to HuggingFace dataset format.

**Returns:**
- `True` if successful, `False` otherwise

**Behavior:**
- Runs `convert_journal_club_dataset.py`
- Converts JSONL to HF dataset
- Creates train/test split (90/10)
- Saves to `HF_DATASET_PATH`

**Example:**
```python
success = convert_to_hf_dataset()
if success:
    print("Dataset converted")
```

### `trigger_vlab2_training()`

Trigger VLAB2's training pipeline.

**Returns:**
- `True` if successful, `False` otherwise

**Behavior:**
- Checks if VLAB2 exists
- Runs VLAB2's `train_lora.py` with journal club config
- Uses journal club training config
- Logs result

**Example:**
```python
success = trigger_vlab2_training()
if success:
    print("Training completed")
```

### `merge_lora_weights()`

Merge LoRA adapter weights into base model.

**Returns:**
- `True` if successful, `False` otherwise

**Behavior:**
- Runs VLAB2's `merge_lora.py`
- Merges adapter from `OUTPUT_MODEL_PATH`
- Saves merged model to `MERGED_MODEL_PATH`
- Logs result

**Example:**
```python
success = merge_lora_weights()
if success:
    print("Model merged")
```

### `mark_papers_as_trained(memory)`

Mark papers as trained in memory.

**Parameters:**
- `memory`: JournalClubMemory instance

**Behavior:**
- Increments training version
- Updates last training date
- Saves memory

**Example:**
```python
mark_papers_as_trained(memory)
```

### `run_training_pipeline(memory)`

Run the complete training pipeline.

**Parameters:**
- `memory`: JournalClubMemory instance (optional, will create if not provided)

**Returns:**
- `True` if pipeline completed successfully, `False` otherwise

**Pipeline Steps:**
1. Check training threshold
2. Collect training data
3. Convert to HuggingFace dataset
4. Trigger VLAB2 training
5. Merge LoRA weights
6. Mark papers as trained

**Behavior:**
- Logs each step
- Stops if any step fails
- Returns success/failure status

**Example:**
```python
from core.training_trigger import run_training_pipeline
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
success = run_training_pipeline(memory)
if success:
    print("Training pipeline completed")
```

### `check_and_trigger_training()`

Check if training should be triggered and run if needed.

**Returns:**
- `True` if training was triggered, `False` otherwise

**Behavior:**
- Creates memory instance
- Checks training version vs paper count
- Triggers training if threshold met
- Returns status

**Example:**
```python
from core.training_trigger import check_and_trigger_training

triggered = check_and_trigger_training()
if triggered:
    print("Training triggered")
```

## Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Check Threshold                                         │
│     ├─ Is JOURNAL_CLUB_LORA_TRAIN=1?                        │
│     └─ Is paper_count >= MIN_TRAIN_PAPERS?                 │
│                                                              │
│  2. Collect Training Data                                   │
│     ├─ Load literature memory                                │
│     ├─ Filter by quality score                               │
│     ├─ Convert papers to instruction-tuning format          │
│     └─ Save to JSONL files                                  │
│                                                              │
│  3. Convert to HuggingFace Dataset                          │
│     ├─ Load JSONL files                                     │
│     ├─ Apply chat template                                  │
│     ├─ Create train/test split (90/10)                      │
│     └─ Save to disk                                         │
│                                                              │
│  4. Trigger VLAB2 Training                                  │
│     ├─ Run VLAB2's train_lora.py                           │
│     ├─ Use journal club training config                     │
│     ├─ Train LoRA adapter                                   │
│     └─ Save adapter to OUTPUT_MODEL_PATH                    │
│                                                              │
│  5. Merge LoRA Weights                                      │
│     ├─ Run VLAB2's merge_lora.py                            │
│     ├─ Merge adapter into base model                        │
│     └─ Save to MERGED_MODEL_PATH                            │
│                                                              │
│  6. Mark Papers as Trained                                  │
│     ├─ Increment training version                           │
│     ├─ Update last training date                            │
│     └─ Save memory                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Usage Patterns

### Manual Trigger
```python
from core.training_trigger import run_training_pipeline
from core.literature_memory import JournalClubMemory

memory = JournalClubMemory()
success = run_training_pipeline(memory)
```

### Automatic Check
```python
from core.training_trigger import check_and_trigger_training

triggered = check_and_trigger_training()
```

### Individual Steps
```python
from core.training_trigger import (
    collect_training_data,
    convert_to_hf_dataset,
    trigger_vlab2_training,
    merge_lora_weights
)

memory = JournalClubMemory()

# Step 1: Collect data
count = collect_training_data(memory)

# Step 2: Convert dataset
convert_to_hf_dataset()

# Step 3: Train
trigger_vlab2_training()

# Step 4: Merge
merge_lora_weights()
```

## Error Handling

- Threshold check fails gracefully (logs and returns False)
- Data collection errors are caught and logged
- Dataset conversion errors are caught and logged
- Training errors are caught and logged
- Merge errors are caught and logged
- Pipeline stops on first failure

## Training Version Tracking

The memory tracks training versions to avoid retraining on the same papers:

```python
memory.memory["training_version"] = 1
memory.memory["last_training_date"] = "2026-07-22T12:00:00Z"
```

## Integration with VLAB2

The training trigger integrates with VLAB2's training infrastructure:

| Component | VLAB2 Component | Purpose |
|-----------|---------------|---------|
| Training | `train_lora.py` | LoRA fine-tuning |
| Merging | `merge_lora.py` | Model merging |
| Config | `training_config.yaml` | Training hyperparameters |

## Performance Considerations

- Training is I/O and compute intensive
- Data collection scales with number of papers
- Dataset conversion is O(n) where n is number of examples
- Training time depends on dataset size and GPU
- Merging is O(model_size)

## Monitoring

### Logging
Training trigger logs at INFO level:
- Threshold check status
- Data collection count
- Dataset conversion status
- Training progress
- Merge status
- Training version updates

### Statistics
Track:
- Training version
- Last training date
- Papers per training version
- Examples per training version

## Troubleshooting

### Training not triggering
- Check `JOURNAL_CLUB_LORA_TRAIN=1`
- Verify paper count meets threshold
- Check quality score threshold

### Data collection fails
- Check memory has analyzed papers
- Verify quality scores meet threshold
- Check output directory permissions

### Dataset conversion fails
- Check JSONL files exist
- Verify JSONL format is valid
- Check output directory permissions

### Training fails
- Check VLAB2 directory exists
- Verify VLAB2 training scripts exist
- Check training config file
- Verify GPU availability

### Merge fails
- Check adapter exists
- Verify VLAB2 merge script exists
- Check output directory permissions
- Verify disk space

## Best Practices

1. **Quality First**: Set appropriate quality score threshold to ensure high-quality training data
2. **Incremental Training**: Retrain periodically as more papers are ingested
3. **Version Tracking**: Monitor training versions to track model evolution
4. **Validation**: Test fine-tuned model before enabling in production
5. **Backup**: Keep backups of previous model versions for rollback
