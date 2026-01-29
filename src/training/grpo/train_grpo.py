#!/usr/bin/env python3
"""
GRPO training script for arithmetic countdown problems.

This script trains a language model using GRPO (Group Relative Policy Optimization)
to solve arithmetic problems with proper reasoning and formatting.
"""

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from src.utils.dataset import load_csv_dataset
from transformers import AutoModelForCausalLM, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer

from src.utils.rewards import (
    mathematical_correctness_reward_function,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("grpo_training")


def load_train_dataset(
    dataset_csv: str, max_rows: int = 2000, seed: int = 42
) -> Dataset:
    """
    Load, shuffle, and subsample the training dataset.

    Args:
        dataset_csv: Absolute path to the dataset CSV file
        max_rows: Maximum number of rows to select for training
        seed: Seed for dataset shuffling

    Returns:
        Dataset: A datasets.Dataset ready for GRPO training
    """
    raw_dataset: Dataset = load_csv_dataset(dataset_csv)
    raw_dataset = raw_dataset.shuffle(seed=seed)
    train_dataset = raw_dataset.select(range(min(max_rows, len(raw_dataset))))
    logger.info("Train rows: %d", len(train_dataset))
    return train_dataset


def create_lora_model(model_id: str, device_map: str = "cuda") -> PreTrainedModel:
    """
    Create a base causal LM and wrap it with LoRA adapters.

    Args:
        model_id: Hugging Face model identifier to load as the base model
        device_map: Device mapping strategy for model loading

    Returns:
        PreTrainedModel: A transformers.PreTrainedModel with LoRA adapters applied
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("Model with LoRA ready")
    return model


def create_grpo_config(
    output_dir: str,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_completion_length: int = 512,
    num_generations: int = 16,
    temperature: float = 1.0,
    save_steps: int = 50,
    logging_steps: int = 1,
    max_prompt_length: int = 4096,
) -> GRPOConfig:
    """
    Create GRPO training configuration.

    Args:
        output_dir: Directory where checkpoints and logs will be written
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        max_completion_length: Maximum length for completions
        num_generations: Number of generations per prompt
        temperature: Sampling temperature
        save_steps: Steps between model saves
        logging_steps: Steps between log outputs
        max_prompt_length: Maximum length for input prompts

    Returns:
        GRPOConfig: A configured trl.GRPOConfig instance
    """
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        bf16=True,
        per_device_train_batch_size=per_device_train_batch_size,
        temperature=temperature,
        # Preprocessing controls
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        # Logging and saving
        report_to=["tensorboard"],
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
    )


def create_trainer(
    model: PreTrainedModel,
    train_dataset: Dataset,
    args: GRPOConfig,
) -> GRPOTrainer:
    """
    Construct a GRPOTrainer with arithmetic-specific reward functions.

    Args:
        model: The LoRA-wrapped pretrained model to train
        train_dataset: The dataset to use for training
        args: The GRPO configuration

    Returns:
        GRPOTrainer: An initialized trl.GRPOTrainer instance
    """
    reward_funcs: list[Callable[..., list[float]]] = [
        mathematical_correctness_reward_function,
    ]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
    )
    return trainer


def train_and_save(trainer: GRPOTrainer, output_dir: str) -> None:
    """
    Run training and save the final model to disk.

    Args:
        trainer: The configured GRPO trainer instance
        output_dir: Output directory to save the trained model

    Returns:
        None
    """
    train_result = trainer.train()
    logger.info("Training complete: %s", str(train_result))
    trainer.save_model(output_dir)
    logger.info("Saved to %s", output_dir)


def main() -> None:
    """
    Run the full GRPO training workflow with command-line arguments.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using GRPO for arithmetic countdown problems"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Path to the training dataset CSV file",
    )
    parser.add_argument(
        "--max_rows", type=int, default=2000, help="Maximum number of training samples"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset shuffling"
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--device_map", type=str, default="x", help="Device mapping strategy"
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Maximum completion length",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--save_steps", type=int, default=50, help="Steps between model saves"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Steps between log outputs"
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=4096,
        help="Maximum length for input prompts",
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.dataset_csv).exists():
        logger.error("Dataset CSV file does not exist: %s", args.dataset_csv)
        return

    if args.max_rows <= 0:
        logger.error("max_rows must be positive")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output dir: %s", args.output_dir)

    # Load dataset
    train_dataset = load_train_dataset(args.dataset_csv, args.max_rows, args.seed)

    # Create model
    model = create_lora_model(args.model_id, args.device_map)

    # Create training configuration
    training_args = create_grpo_config(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_prompt_length=args.max_prompt_length,
    )

    # Create trainer
    trainer = create_trainer(
        model=model, train_dataset=train_dataset, args=training_args
    )

    # Train and save
    train_and_save(trainer=trainer, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
