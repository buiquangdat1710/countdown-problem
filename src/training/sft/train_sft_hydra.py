#!/usr/bin/env python3
"""
SFT training script for arithmetic countdown problems using Hydra configuration.

This script trains a language model using SFT (Supervised Fine-Tuning)
to solve arithmetic problems with proper reasoning and formatting.
"""

import logging
import os
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import SFTConfig, SFTTrainer

from src.dataset.sft import (
    load_csv_dataset_sft,
    map_problem_description_to_conversation_sft,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sft_training")


def load_train_dataset(cfg: DictConfig) -> Dataset:
    """
    Load, shuffle, and subsample the training dataset.

    Args:
        cfg: Dataset configuration

    Returns:
        Dataset: A datasets.Dataset ready for SFT training
    """
    raw_dataset: Dataset = load_csv_dataset_sft(
        cfg.file_path, map_problem_description_to_conversation_sft
    )
    raw_dataset = raw_dataset.shuffle(seed=cfg.seed)
    train_dataset = raw_dataset.select(range(min(cfg.max_rows, len(raw_dataset))))
    logger.info("Train rows: %d", len(train_dataset))
    return train_dataset


def create_lora_model(cfg: DictConfig) -> PreTrainedModel:
    """
    Create a base causal LM and wrap it with LoRA adapters.

    Args:
        cfg: Model configuration

    Returns:
        PreTrainedModel: A transformers.PreTrainedModel with LoRA adapters applied
    """
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map=cfg.device_map,
    )

    # Convert Hydra config objects to plain Python objects for JSON serialization
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=OmegaConf.to_container(cfg.lora.target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("Model with LoRA ready")
    return model


def create_sft_config(cfg: DictConfig, output_dir: str) -> SFTConfig:
    """
    Create SFT training configuration from Hydra config.

    Args:
        cfg: Training configuration
        output_dir: Directory where checkpoints and logs will be written

    Returns:
        SFTConfig: A configured trl.SFTConfig instance
    """
    return SFTConfig(
        output_dir=output_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        remove_unused_columns=cfg.remove_unused_columns,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        # SFT-specific parameters
        max_length=cfg.max_length,
        # Logging and saving
        report_to=cfg.report_to,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
    )


def create_trainer(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    args: SFTConfig,
) -> SFTTrainer:
    """
    Construct an SFTTrainer for supervised fine-tuning.

    Args:
        model: The LoRA-wrapped pretrained model to train
        tokenizer: The tokenizer for the model
        train_dataset: The dataset to use for training
        args: The SFT configuration

    Returns:
        SFTTrainer: An initialized trl.SFTTrainer instance
    """
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    return trainer


def train_and_save(
    trainer: SFTTrainer, model: PreTrainedModel, output_dir: str
) -> None:
    """
    Run training and save the final model to disk.

    Args:
        trainer: The configured SFT trainer instance
        model: The model to merge and save
        output_dir: Output directory to save the trained model

    Returns:
        None
    """
    train_result = trainer.train()
    logger.info("Training complete: %s", str(train_result))
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    logger.info("Saved to %s", output_dir)


@hydra.main(version_base=None, config_path="../../config/sft", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run the full SFT training workflow with Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        None
    """
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Validate dataset file exists
    if not Path(cfg.dataset.file_path).exists():
        logger.error("Dataset CSV file does not exist: %s", cfg.dataset.file_path)
        return

    if cfg.dataset.max_rows <= 0:
        logger.error("max_rows must be positive")
        return

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info("Output dir: %s", cfg.output_dir)

    # Load dataset
    train_dataset = load_train_dataset(cfg.dataset)

    # Create model and tokenizer
    model = create_lora_model(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)

    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create training configuration
    training_args = create_sft_config(cfg.training, cfg.output_dir)

    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    # Train and save
    train_and_save(trainer=trainer, model=model, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
