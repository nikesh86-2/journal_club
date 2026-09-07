import subprocess
import os
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("journal_club.training_trigger")

TRAINING_DATA_DIR = Path("training/journal_club_data")
HF_DATASET_PATH = Path("training/journal_club_hf_dataset")
OUTPUT_MODEL_PATH = Path("training/journal_club_output")
MERGED_MODEL_PATH = Path("training/journal_club_merged_model")


def check_training_threshold(memory):
    """Check if training threshold is met."""
    lora_train = os.getenv("JOURNAL_CLUB_LORA_TRAIN", "0").strip() == "1"
    min_train_papers = int(os.getenv("JOURNAL_CLUB_MIN_TRAIN_PAPERS", "200"))

    if not lora_train:
        log.info("Training disabled (JOURNAL_CLUB_LORA_TRAIN=0)")
        return False

    total_papers = memory.get_statistics().get("total_papers", 0)
    if total_papers < min_train_papers:
        log.info(f"Training threshold not met: {total_papers} < {min_train_papers}")
        return False

    log.info(f"Training threshold met: {total_papers} >= {min_train_papers}")
    return True


def collect_training_data(memory):
    """Collect training data from literature memory."""
    from core.training_data_collector import collect_training_data_from_memory
    
    count = collect_training_data_from_memory()
    log.info(f"Collected {count} training examples")
    return count


def convert_to_hf_dataset():
    """Convert JSONL data to HuggingFace dataset format."""
    convert_script = Path("training/convert_journal_club_dataset.py")
    if not convert_script.exists():
        log.error(f"Dataset converter not found: {convert_script}")
        return False
    
    try:
        subprocess.run(["python", str(convert_script)], check=True)
        log.info("Dataset conversion completed")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Dataset conversion failed: {e}")
        return False


def trigger_training():
    """Trigger the training pipeline."""
    train_script = Path("core/train_lora.py")
    if not train_script.exists():
        log.error(f"Training script not found: {train_script}")
        return False
    
    try:
        subprocess.run(["python", str(train_script)], check=True)
        log.info("Training completed")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Training failed: {e}")
        return False


def merge_lora_weights():
    """Merge LoRA adapter weights into base model."""
    merge_script = Path("core/merge_lora.py")
    if not merge_script.exists():
        log.error(f"Merge script not found: {merge_script}")
        return False
    
    try:
        subprocess.run(["python", str(merge_script)], check=True)
        log.info("Model merge completed")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Model merge failed: {e}")
        return False


def mark_papers_as_trained(memory):
    """Mark papers as trained in memory."""
    training_version = int(memory.get_metadata("training_version", 0) or 0) + 1
    memory.set_metadata("training_version", training_version)
    memory.set_metadata("last_training_date", datetime.utcnow().isoformat() + "Z")
    memory.save()
    log.info(f"Marked papers as trained (version {training_version})")


def run_training_pipeline(memory=None):
    """Run the complete training pipeline."""
    if memory is None:
        from core.literature_memory import JournalClubMemory
        memory = JournalClubMemory()
    
    # Step 1: Check threshold
    if not check_training_threshold(memory):
        return False
    
    # Step 2: Collect training data
    if collect_training_data(memory) == 0:
        log.warning("No training data collected")
        return False
    
    # Step 3: Convert to HuggingFace dataset
    if not convert_to_hf_dataset():
        return False
    
    # Step 4: Trigger training
    if not trigger_training():
        return False
    
    # Step 5: Merge LoRA weights
    if not merge_lora_weights():
        return False
    
    # Step 6: Mark papers as trained
    mark_papers_as_trained(memory)
    
    return True


def check_and_trigger_training():
    """Check if training should be triggered and run if needed."""
    from core.literature_memory import JournalClubMemory
    
    memory = JournalClubMemory()
    return run_training_pipeline(memory)