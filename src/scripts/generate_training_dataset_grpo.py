#!/usr/bin/env python3
"""
Script to generate a CSV file containing arithmetic problems for training data.

This script uses the arithmetic utilities to generate problems and creates a CSV
with columns: id, problem_description, correct_answer, num1, num2, num3, num4.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from src.utils.arithmetics import (
    ArithmeticProblemDescriptionGenerator,
    ArithmeticProblemGenerator,
    Mode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_training_data(num_problems: int) -> list[dict[str, Any]]:
    """
    Generate training data with arithmetic problems.

    Args:
        num_problems: Number of problems to generate

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing training data
    """
    problem_generator = ArithmeticProblemGenerator(mode=Mode.MUL_DIV)
    description_generator = ArithmeticProblemDescriptionGenerator()

    training_data = []
    generated_count = 0
    attempts = 0
    max_total_attempts = num_problems * 10  # Allow more attempts than problems

    logger.info(f"Starting generation of {num_problems} problems...")

    while generated_count < num_problems and attempts < max_total_attempts:
        attempts += 1

        # Generate a problem
        problem = problem_generator.generate_problem()

        if problem is None:
            continue

        # Generate description
        problem_description, correct_answer = (
            description_generator.generate_description(problem)
        )

        # Create training data entry
        training_entry = {
            "id": generated_count + 1,
            "problem_description": problem_description,
            "correct_answer": correct_answer,
            "num1": problem.num_1,
            "num2": problem.num_2,
            "num3": problem.num_3,
            "num4": problem.num_4,
        }

        training_data.append(training_entry)
        generated_count += 1

        if generated_count % 100 == 0:
            logger.info(f"Generated {generated_count} problems...")

    if generated_count < num_problems:
        logger.warning(
            f"Only generated {generated_count} out of {num_problems} requested problems after {attempts} attempts"
        )
    else:
        logger.info(
            f"Successfully generated {generated_count} problems in {attempts} attempts"
        )

    return training_data


def save_to_csv(training_data: list[dict[str, Any]], output_file: Path) -> None:
    """
    Save training data to a CSV file.

    Args:
        training_data: List of training data dictionaries
        output_file: Path to the output CSV file
    """
    if not training_data:
        logger.error("No training data to save")
        return

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(training_data)} problems to {output_file}")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id",
            "problem_description",
            "correct_answer",
            "num1",
            "num2",
            "num3",
            "num4",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data
        for entry in training_data:
            writer.writerow(entry)

    logger.info(f"Successfully saved training data to {output_file}")


def main() -> None:
    """
    Main function to handle command line arguments and orchestrate the generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate arithmetic problems for training data in CSV format"
    )

    parser.add_argument(
        "--num_problems", type=int, required=True, help="Number of problems to generate"
    )

    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output CSV file"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_problems <= 0:
        logger.error("Number of problems must be positive")
        return

    output_path = Path(args.output_file)

    # Generate training data
    training_data = generate_training_data(args.num_problems)

    if not training_data:
        logger.error("Failed to generate any training data")
        return

    # Save to CSV
    save_to_csv(training_data, output_path)


if __name__ == "__main__":
    main()
