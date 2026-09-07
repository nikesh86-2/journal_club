"""
QLoRA fine-tuning for Journal Club.

Designed for low-VRAM GPUs such as RTX 3050 6 GB.

Uses:
    - 4-bit NF4 quantisation via bitsandbytes
    - LoRA adapters
    - gradient checkpointing
    - paged AdamW 8-bit
    - BF16 compute where supported

Usage:
    python core/train_lora.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Ensure repo root is on sys.path so `from core...` works when this file is
# run directly (e.g. `python core/train_lora.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import resolve_env

log = logging.getLogger("journal_club.training")

CONFIG_PATH = Path("training/journal_club_training_config.yaml")
HF_DATASET_PATH = Path("training/journal_club_hf_dataset/hf_disk")
OUTPUT_DIR = Path("training/journal_club_output")
LOGS_DIR = Path("training/logs")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return resolve_env(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "training.log"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(config: dict):
    """Load model for LoRA/QLoRA training with 4-bit quantization."""
    # Use student model if configured, otherwise use base model
    model_name = config.get("student_model_name", config["model_name"])
    is_student_model = "student_model_name" in config

    log.info(f"Loading model: {model_name}")
    if is_student_model:
        log.info("Using student model for training (smaller model for VRAM efficiency)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("Set pad_token to eos_token")

    # Use 4-bit quantization via bitsandbytes for VRAM efficiency
    log.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    return model, tokenizer

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def setup_lora(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """Prepare GPTQ model and attach LoRA adapters."""
    lora_config = config.get("lora", {})

    log.info("Preparing quantized model for LoRA training...")

    # Required preparation for quantized PEFT training.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_cfg = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    log.info(
        f"LoRA config: r={lora_cfg.r}, "
        f"alpha={lora_cfg.lora_alpha}, "
        f"target_modules={lora_cfg.target_modules}"
    )

    model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset():
    if not HF_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {HF_DATASET_PATH}. "
            "Run training/convert_journal_club_dataset.py first."
        )

    log.info("Loading dataset from %s", HF_DATASET_PATH)

    dataset = load_from_disk(str(HF_DATASET_PATH))

    log.info("Dataset loaded: %s examples", len(dataset))

    return dataset


def tokenize_function(examples, tokenizer, max_length):
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = [
        ids.copy() for ids in model_inputs["input_ids"]
    ]

    return model_inputs


def prepare_dataset(dataset, tokenizer, max_length):
    log.info("Tokenizing dataset...")
    log.info("Maximum sequence length: %s", max_length)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(
            x,
            tokenizer,
            max_length,
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    return tokenized_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()

    log.info("========================================")
    log.info("        JOURNAL CLUB QLoRA TRAINING")
    log.info("========================================")

    config = load_config()

    log.info("Configuration loaded from %s", CONFIG_PATH)

    # ---------------------------------------------------------------
    # CUDA diagnostics
    # ---------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    log.info(
        "CUDA device: %s",
        torch.cuda.get_device_name(0),
    )

    free, total = torch.cuda.mem_get_info()

    log.info(
        "GPU memory before model load: %.2f GB free / %.2f GB total",
        free / 1024**3,
        total / 1024**3,
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model, tokenizer = load_model_and_tokenizer(config)

    # ---------------------------------------------------------------
    # LoRA
    # ---------------------------------------------------------------

    model = setup_lora(model, config)

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    dataset = load_dataset()

    # IMPORTANT:
    #
    # 4096 tokens can still be expensive on a 6 GB GPU even with QLoRA.
    #
    # Start at 2048 unless you know you need 4096.
    max_length = config.get("max_length", 2048)

    tokenized_dataset = prepare_dataset(
        dataset,
        tokenizer,
        max_length,
    )

    if "train" not in tokenized_dataset:
        split = tokenized_dataset.train_test_split(
            test_size=0.1,
            seed=42,
        )

        train_dataset = split["train"]
        eval_dataset = split["test"]

    else:
        train_dataset = tokenized_dataset["train"]

        if "test" in tokenized_dataset:
            eval_dataset = tokenized_dataset["test"]
        elif "validation" in tokenized_dataset:
            eval_dataset = tokenized_dataset["validation"]
        else:
            split = train_dataset.train_test_split(
                test_size=0.1,
                seed=42,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]

    log.info("Train examples: %s", len(train_dataset))
    log.info("Eval examples: %s", len(eval_dataset))

    # ---------------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------------

    training_config = config.get("training", {})

    # Eval / best-model selection wired from the YAML config
    eval_strategy = training_config.get("eval_strategy", "no")
    eval_steps = int(training_config.get("eval_steps", 250))
    save_best = bool(training_config.get("save_best_model", False))
    save_steps = int(training_config.get("save_steps", 100))

    if save_best and eval_strategy in ("no", None, ""):
        log.warning(
            "save_best_model=true but no eval_strategy configured; "
            "enabling 'steps' evaluation"
        )
        eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=training_config.get(
            "epochs",
            1,
        ),

        # RTX 3050 6 GB:
        # keep physical batch size at 1.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps",
            8,
        ),

        learning_rate=training_config.get(
            "learning_rate",
            2e-5,
        ),

        warmup_steps=100,

        logging_steps=training_config.get(
            "logging_steps",
            5,
        ),

        # -----------------------------------------------------------
        # Evaluation / best-model selection
        # -----------------------------------------------------------

        eval_strategy=eval_strategy,
        eval_steps=eval_steps,

        # When keeping the best checkpoint, save_strategy must match
        # eval_strategy; otherwise save on a fixed step cadence.
        save_strategy=eval_strategy if save_best else "steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=save_best,
        metric_for_best_model=training_config.get(
            "metric_for_best_model",
            "eval_loss",
        ),
        greater_is_better=False,

        # -----------------------------------------------------------
        # Low VRAM settings
        # -----------------------------------------------------------

        bf16=True,

        gradient_checkpointing=True,

        # 8-bit paged optimizer dramatically reduces optimizer memory.
        optim="paged_adamw_8bit",

        # Avoid unnecessary GPU memory spikes.
        max_grad_norm=1.0,

        # -----------------------------------------------------------
        # Logging / saving
        # -----------------------------------------------------------

        report_to="none",

        # Don't retain unnecessary tensors.
        remove_unused_columns=True,

        # Useful with gradient checkpointing.
        ddp_find_unused_parameters=False,
    )

    log.info("Training arguments:")
    log.info("%s", training_args)

    # ---------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        # Transformers 5.x supports processing_class.
        processing_class=tokenizer,
    )

    # ---------------------------------------------------------------
    # Final GPU diagnostics
    # ---------------------------------------------------------------

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3

    log.info(
        "GPU memory before training: %.2f GB allocated / %.2f GB reserved",
        allocated,
        reserved,
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    log.info("Starting QLoRA training...")

    trainer.train()

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    log.info("Saving LoRA adapter to %s", OUTPUT_DIR)

    trainer.save_model()

    tokenizer.save_pretrained(OUTPUT_DIR)

    log.info("========================================")
    log.info("       QLoRA TRAINING COMPLETE")
    log.info("========================================")


if __name__ == "__main__":
    main()