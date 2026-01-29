#!/usr/bin/env python3
"""
Script to calculate accuracy of a trained GRPO model on arithmetic countdown problems.

This script loads a CSV file with problem data, performs inference using the trained model,
and calculates the accuracy by comparing predicted answers with correct answers.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.inference import GRPOModelInference

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("calculate_accuracy")


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data with the expected format.

    Expected columns: id, problem_description, correct_answer, num1, num2, num3, num4

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_columns = [
        "id",
        "problem_description",
        "correct_answer",
        "num1",
        "num2",
        "num3",
        "num4",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Loaded {len(df)} problems from {csv_path}")
    return df


def safe_eval_expression(expression: str) -> tuple[float | None, bool]:
    """
    Safely evaluate an arithmetic expression.

    Args:
        expression: The arithmetic expression to evaluate

    Returns:
        Tuple of (result, is_valid)
    """
    if not expression or not expression.strip():
        return None, False

    # Replace 'x' with '*' for evaluation if present
    normalized = expression.replace("x", "*").replace("X", "*")

    # Basic validation - only allow numbers, operators, spaces, and parentheses
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in normalized):
        return None, False

    try:
        result = eval(normalized)
        return result, True
    except (SyntaxError, ValueError, ZeroDivisionError, NameError):
        return None, False


def check_numbers_usage(expression: str, required_numbers: list[int]) -> bool:
    """
    Check if the expression uses exactly the required numbers.

    Args:
        expression: The arithmetic expression to check
        required_numbers: List of numbers that should be used exactly once each

    Returns:
        True if expression uses all required numbers exactly once, False otherwise
    """
    if not expression or not expression.strip():
        return False

    # Extract all numbers from the expression
    numbers_in_expression = re.findall(r"\b\d+\b", expression)

    # Convert to integers
    try:
        numbers_in_expression = [int(num) for num in numbers_in_expression]
    except ValueError:
        return False

    # Sort both lists for comparison
    required_sorted = sorted(required_numbers)
    found_sorted = sorted(numbers_in_expression)

    return required_sorted == found_sorted


def evaluate_prediction(
    predicted_answer: str, correct_answer: int, nums: list[int]
) -> dict:
    """
    Evaluate a single prediction against the correct answer.

    Args:
        predicted_answer: The model's predicted arithmetic expression
        correct_answer: The correct integer result
        nums: List of four numbers used in the problem

    Returns:
        Dictionary with evaluation results
    """
    result = {
        "predicted_answer": predicted_answer,
        "correct_answer": correct_answer,
        "is_correct": False,
        "is_valid_format": False,
        "uses_all_numbers": False,
        "predicted_result": None,
        "correct_result": correct_answer,
    }

    # Evaluate predicted answer
    predicted_result, is_valid_predicted = safe_eval_expression(predicted_answer)
    result["predicted_result"] = predicted_result
    result["is_valid_format"] = is_valid_predicted

    # Check if all required numbers are used
    uses_all_numbers = check_numbers_usage(predicted_answer, nums)
    result["uses_all_numbers"] = uses_all_numbers

    # Log predicted and correct results
    logger.info(
        f"Answered: {predicted_answer} - Predicted result: {predicted_result} - Correct result: {correct_answer} - Uses all numbers: {uses_all_numbers}"
    )

    # Check if prediction is correct (must be valid format, use all numbers, and have correct result)
    if is_valid_predicted and predicted_result is not None and uses_all_numbers:
        result["is_correct"] = abs(predicted_result - correct_answer) < 1e-6

    return result


def calculate_accuracy(
    csv_path: str,
    sft_model_path: str | None,
    grpo_model_path: str | None,
    base_model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    max_samples: int | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Calculate accuracy of the model on the given dataset.

    Args:
        csv_path: Path to the CSV file with test data
        sft_model_path: Path to the SFT model
        grpo_model_path: Path to the GRPO model
        base_model_id: Base model identifier
        device: Device to run inference on
        dtype: Data type for the model
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_samples: Maximum number of samples to evaluate (None for all)
        output_path: Path to save detailed results (optional)

    Returns:
        Dictionary with accuracy metrics
    """
    # Load data
    df = load_csv_data(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)
        logger.info(f"Limiting evaluation to {max_samples} samples")

    # Initialize model
    logger.info("Loading model...")
    model_inference = GRPOModelInference(
        sft_model_path=sft_model_path,
        grpo_model_path=grpo_model_path,
        base_model_id=base_model_id,
        device=device,
        dtype=dtype,
    )

    # Evaluate each problem
    results = []
    correct_predictions = 0
    valid_format_predictions = 0
    uses_all_numbers_predictions = 0

    logger.info("Starting evaluation...")
    pbar = tqdm(df.iterrows(), total=len(df), desc="Evaluating")

    for idx, (_, row) in enumerate(pbar):
        # Perform inference
        response, extracted_answer, _ = model_inference.solve_problem(
            problem_description=row["problem_description"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Evaluate prediction
        nums = [row["num1"], row["num2"], row["num3"], row["num4"]]
        evaluation = evaluate_prediction(
            predicted_answer=extracted_answer,
            correct_answer=row["correct_answer"],
            nums=nums,
        )

        # Add metadata
        evaluation.update(
            {
                "id": row["id"],
                "problem_description": row["problem_description"],
                "full_response": response,
                "nums": nums,
            }
        )

        results.append(evaluation)

        # Update counters
        if evaluation["is_correct"]:
            correct_predictions += 1
        if evaluation["is_valid_format"]:
            valid_format_predictions += 1
        if evaluation["uses_all_numbers"]:
            uses_all_numbers_predictions += 1

        # Update progress bar with intermediate results
        current_accuracy = correct_predictions / (idx + 1) if (idx + 1) > 0 else 0
        current_valid_rate = (
            valid_format_predictions / (idx + 1) if (idx + 1) > 0 else 0
        )
        pbar.set_postfix(
            {
                "Acc": f"{current_accuracy:.3f}",
                "Valid": f"{current_valid_rate:.3f}",
                "Correct": f"{correct_predictions}/{idx + 1}",
            }
        )

    # Calculate metrics
    total_samples = len(results)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    valid_format_rate = (
        valid_format_predictions / total_samples if total_samples > 0 else 0
    )
    uses_all_numbers_rate = (
        uses_all_numbers_predictions / total_samples if total_samples > 0 else 0
    )

    metrics = {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "valid_format_predictions": valid_format_predictions,
        "uses_all_numbers_predictions": uses_all_numbers_predictions,
        "accuracy": accuracy,
        "valid_format_rate": valid_format_rate,
        "uses_all_numbers_rate": uses_all_numbers_rate,
    }

    # Log results
    logger.info("Evaluation completed!")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Valid format predictions: {valid_format_predictions}")
    logger.info(f"Uses all numbers predictions: {uses_all_numbers_predictions}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    logger.info(
        f"Valid format rate: {valid_format_rate:.4f} ({valid_format_rate * 100:.2f}%)"
    )
    logger.info(
        f"Uses all numbers rate: {uses_all_numbers_rate:.4f} ({uses_all_numbers_rate * 100:.2f}%)"
    )

    # Save detailed results if requested
    if output_path:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Detailed results saved to {output_path}")

    return metrics


def main():
    """Main function to run the accuracy calculation script."""
    parser = argparse.ArgumentParser(
        description="Calculate accuracy of GRPO model on arithmetic countdown problems"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        default="data/grpo/test.csv",
        help="Path to CSV file with test data",
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="models/sft/",
        help="Path to SFT model directory",
    )
    parser.add_argument(
        "--grpo_model_path",
        type=str,
        default="models/grpo/",
        help="Path to GRPO model directory",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base model identifier",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to run inference on"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save detailed results CSV",
    )
    parser.add_argument(
        "--no_sft",
        action="store_true",
        help="Skip loading the SFT model (use only base model)",
    )
    parser.add_argument(
        "--no_grpo",
        action="store_true",
        help="Skip loading the GRPO model (use only SFT model)",
    )

    args = parser.parse_args()

    # Convert dtype
    dtype = torch.float16

    # Calculate accuracy
    metrics = calculate_accuracy(
        csv_path=args.csv_path,
        sft_model_path=args.sft_model_path if not args.no_sft else None,
        grpo_model_path=args.grpo_model_path if not args.no_grpo else None,
        base_model_id=args.base_model_id,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        output_path=args.output_path,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(
        f"Valid Format Rate: {metrics['valid_format_rate']:.4f} ({metrics['valid_format_rate'] * 100:.2f}%)"
    )
    print(
        f"Uses All Numbers Rate: {metrics['uses_all_numbers_rate']:.4f} ({metrics['uses_all_numbers_rate'] * 100:.2f}%)"
    )
    print(
        f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_samples']}"
    )
    print(
        f"Valid Format Predictions: {metrics['valid_format_predictions']}/{metrics['total_samples']}"
    )
    print(
        f"Uses All Numbers Predictions: {metrics['uses_all_numbers_predictions']}/{metrics['total_samples']}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
