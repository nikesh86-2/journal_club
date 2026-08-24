"""
convert_journal_club_dataset.py

Converts journal club JSONL training data into HuggingFace dataset format.

Input:
  training/journal_club_data/*.jsonl

Output:
  training/journal_club_hf_dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("journal_club.training")


INPUT_DIR = Path("training/journal_club_data")
SAVE_PATH = Path("training/journal_club_hf_dataset")


SYSTEM_PROMPT = (
    "You are a scientific literature analysis assistant for a journal club. "
    "You specialize in analyzing research papers, identifying gaps in methodology, "
    "generating critiques, scoring paper quality, and recommending related reading. "
    "Provide clear, concise, and scientifically accurate responses."
)


def render_instruction_example(item: dict) -> str:
    """Render an instruction-tuning example in chat template format."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n\nINPUT:\n{input_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output_text}\n"
        "<|im_end|>\n"
    )


def render_chat_example(item: dict) -> str:
    """Render a chat example in chat template format."""
    parts = []

    for msg in item.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")

    return "\n".join(parts) + "\n"


def load_jsonl_files(input_dir: Path) -> list:
    """Load all JSONL files from the input directory."""
    rows = []
    
    if not input_dir.exists():
        log.warning("Input directory not found: %s", input_dir)
        return rows
    
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        log.warning("No JSONL files found in: %s", input_dir)
        return rows
    
    log.info("Loading %d JSONL files from %s", len(jsonl_files), input_dir)
    
    for jsonl_file in jsonl_files:
        log.debug("Loading: %s", jsonl_file.name)
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Failed to parse line %d in %s: %s", line_num, jsonl_file.name, e)
                    continue
                
                if "messages" in item:
                    text = render_chat_example(item)
                else:
                    text = render_instruction_example(item)
                
                if text.strip():
                    rows.append({"text": text})
    
    return rows


def main() -> None:
    """Convert journal club JSONL data to HuggingFace dataset and split JSONL files."""
    import random

    rows = load_jsonl_files(INPUT_DIR)
    
    if not rows:
        log.error("No training examples found. Aborting.")
        return
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(rows)

    split_idx = max(1, int(len(rows) * 0.9))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:] if len(rows) > 1 else rows[:1]

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Save formatted JSONL splits
    train_jsonl = SAVE_PATH / "train.jsonl"
    test_jsonl = SAVE_PATH / "test.jsonl"

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_jsonl, "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Saved formatted JSONL splits to %s and %s", train_jsonl, test_jsonl)

    # Try saving as HuggingFace Dataset if datasets library is installed
    try:
        from datasets import Dataset, DatasetDict
        dataset = DatasetDict({
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows)
        })
        dataset.save_to_disk(str(SAVE_PATH / "hf_disk"))
        log.info("Saved HF Dataset to %s", SAVE_PATH / "hf_disk")
    except ImportError:
        log.info("HuggingFace `datasets` library not installed; JSONL format exported successfully.")

    log.info("Train examples: %d", len(train_rows))
    log.info("Test examples: %d", len(test_rows))
    log.info("Total examples: %d", len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
