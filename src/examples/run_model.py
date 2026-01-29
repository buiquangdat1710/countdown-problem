#!/usr/bin/env python3
"""
Script to use a trained GRPO model for arithmetic countdown problems.

This script loads a model trained with train_grpo_hydra.py and provides
both interactive and batch evaluation modes for solving arithmetic problems.
"""

import logging
import sys
from pathlib import Path

import torch

from src.utils.inference import GRPOModelInference

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_inference")


def main():
    """Main function to run the model inference script."""
    # Create model inference object
    model_inference = GRPOModelInference(
        sft_model_path="models/sft/",
        grpo_model_path="models/grpo/",
        base_model_id="Qwen/Qwen2.5-Math-1.5B",
        device="auto",
        dtype=torch.float16,
    )

    # Solve problem
    response, extracted_answer, is_valid = model_inference.solve_problem(
        problem_description="Your task: Use 53, 3, 47, and 36 exactly once each with only +, -, *, and / operators to create an expression equal to 133.",
        max_new_tokens=1024,
        temperature=1.0,
    )
    logger.info(f"Response: {response}")
    logger.info(f"Extracted Answer: {extracted_answer}")
    logger.info(f"Valid Format: {is_valid}")


if __name__ == "__main__":
    main()
