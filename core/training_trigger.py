"""
training_trigger.py

Monitors paper count and triggers LoRA fine-tuning when threshold is reached.
Integrates with VLAB2's existing training pipeline.
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from .literature_memory import JournalClubMemory
from .training_data_collector import collect_training_data_from_memory

log = logging.getLogger("journal_club.training")


# Configuration
MIN_TRAIN_PAPERS = int(os.getenv("JOURNAL_CLUB_MIN_TRAIN_PAPERS", "200"))
TRAINING_ENABLED = os.getenv("JOURNAL_CLUB_LORA_TRAIN", "0") == "1"
MIN_QUALITY_SCORE = float(os.getenv("JOURNAL_CLUB_MIN_QUALITY_SCORE", "0.5"))

# Paths
VLAB2_PATH = Path(__file__).parents[2] / "VLAB2"
TRAINING_DATA_DIR = Path(__file__).parents[1] / "training" / "journal_club_data"
HF_DATASET_PATH = Path(__file__).parents[1] / "training" / "journal_club_hf_dataset"
OUTPUT_MODEL_PATH = Path(__file__).parents[1] / "training" / "journal_club_output"
MERGED_MODEL_PATH = Path(__file__).parents[1] / "training" / "journal_club_merged_model"


def check_training_threshold(memory: JournalClubMemory) -> bool:
    """Check if training threshold is met."""
    
    if not TRAINING_ENABLED:
        log.info("LoRA training is disabled (JOURNAL_CLUB_LORA_TRAIN=0)")
        return False
    
    stats = memory.get_statistics()
    total_papers = stats["total_papers"]
    
    if total_papers < MIN_TRAIN_PAPERS:
        log.info("Training threshold not met: %d papers (need %d)", total_papers, MIN_TRAIN_PAPERS)
        return False
    
    log.info("Training threshold met: %d papers >= %d", total_papers, MIN_TRAIN_PAPERS)
    return True


def collect_training_data(memory: JournalClubMemory) -> int:
    """Collect training data from literature memory."""
    
    log.info("Collecting training data from literature memory...")
    
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    example_count = collect_training_data_from_memory(
        memory_path=memory.path,
        output_dir=str(TRAINING_DATA_DIR),
        min_quality_score=MIN_QUALITY_SCORE,
    )
    
    log.info("Collected %d training examples", example_count)
    
    if example_count < MIN_TRAIN_PAPERS:
        log.warning("Insufficient training examples: %d (need %d)", example_count, MIN_TRAIN_PAPERS)
        return 0
    
    return example_count


def convert_to_hf_dataset() -> bool:
    """Convert JSONL data to HuggingFace dataset format."""
    
    log.info("Converting to HuggingFace dataset format...")
    
    converter_script = Path(__file__).parents[1] / "training" / "convert_journal_club_dataset.py"
    
    if not converter_script.exists():
        log.error("Dataset converter not found: %s", converter_script)
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(converter_script)],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            log.error("Dataset conversion failed: %s", result.stderr)
            return False
        
        log.info("Dataset conversion successful")
        return True
        
    except Exception as e:
        log.error("Error running dataset converter: %s", e)
        return False


def trigger_vlab2_training() -> bool:
    """Trigger VLAB2's training pipeline with journal club dataset."""
    
    log.info("Triggering VLAB2 training pipeline...")
    
    # Check if VLAB2 training scripts exist
    if not VLAB2_PATH.exists():
        log.error("VLAB2 not found at %s", VLAB2_PATH)
        return False
    
    train_script = VLAB2_PATH / "training" / "train_lora.py"
    
    if not train_script.exists():
        log.error("VLAB2 training script not found: %s", train_script)
        return False
    
    # Create journal club specific training config
    config_path = Path(__file__).parents[1] / "training" / "journal_club_training_config.yaml"
    
    if not config_path.exists():
        log.error("Journal club training config not found: %s", config_path)
        return False
    
    try:
        # Run VLAB2's train_lora.py with journal club config
        result = subprocess.run(
            [sys.executable, str(train_script), "--config", str(config_path)],
            cwd=str(VLAB2_PATH),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            log.error("Training failed: %s", result.stderr)
            return False
        
        log.info("Training successful")
        return True
        
    except Exception as e:
        log.error("Error running training: %s", e)
        return False


def merge_lora_weights() -> bool:
    """Merge LoRA adapter weights into base model."""
    
    log.info("Merging LoRA weights...")
    
    merge_script = VLAB2_PATH / "training" / "merge_lora.py"
    
    if not merge_script.exists():
        log.error("VLAB2 merge script not found: %s", merge_script)
        return False
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(merge_script),
                "--lora-adapter", str(OUTPUT_MODEL_PATH),
                "--output", str(MERGED_MODEL_PATH),
            ],
            cwd=str(VLAB2_PATH),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            log.error("Merge failed: %s", result.stderr)
            return False
        
        log.info("Merge successful")
        return True
        
    except Exception as e:
        log.error("Error merging weights: %s", e)
        return False


def mark_papers_as_trained(memory: JournalClubMemory) -> None:
    """Mark papers as trained in memory to avoid retraining."""
    
    log.info("Marking papers as trained...")
    
    # Add a training version field to memory
    training_version = memory.memory.get("training_version", 0) + 1
    memory.memory["training_version"] = training_version
    memory.memory["last_training_date"] = datetime.utcnow().isoformat()
    
    memory.save()
    
    log.info("Training version: %d", training_version)


def run_training_pipeline(memory: JournalClubMemory | None = None) -> bool:
    """Run the complete training pipeline."""
    
    if memory is None:
        memory = JournalClubMemory()
    
    log.info("=" * 60)
    log.info("Starting Journal Club LoRA Training Pipeline")
    log.info("=" * 60)
    
    # Step 1: Check threshold
    if not check_training_threshold(memory):
        log.info("Training threshold not met, skipping")
        return False
    
    # Step 2: Collect training data
    example_count = collect_training_data(memory)
    if example_count == 0:
        log.error("No training examples collected, aborting")
        return False
    
    # Step 3: Convert to HF dataset
    if not convert_to_hf_dataset():
        log.error("Dataset conversion failed, aborting")
        return False
    
    # Step 4: Trigger training
    if not trigger_vlab2_training():
        log.error("Training failed, aborting")
        return False
    
    # Step 5: Merge weights
    if not merge_lora_weights():
        log.error("Merge failed, aborting")
        return False
    
    # Step 6: Mark papers as trained
    mark_papers_as_trained(memory)
    
    log.info("=" * 60)
    log.info("Training pipeline completed successfully")
    log.info("=" * 60)
    
    return True


def check_and_trigger_training() -> bool:
    """Check if training should be triggered and run if needed."""
    
    memory = JournalClubMemory()
    
    # Check if already trained on current papers
    training_version = memory.memory.get("training_version", 0)
    total_papers = memory.get_statistics()["total_papers"]
    
    log.info("Current training version: %d, Total papers: %d", training_version, total_papers)
    
    # Simple heuristic: if we have enough new papers, trigger training
    # (In production, you'd track which papers were trained on)
    if total_papers >= MIN_TRAIN_PAPERS:
        return run_training_pipeline(memory)
    
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_trigger_training()
