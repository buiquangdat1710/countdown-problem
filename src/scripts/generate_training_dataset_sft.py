#!/usr/bin/env python3
"""
Script to generate a CSV file containing arithmetic problems for SFT training data.

This script uses the arithmetic utilities to generate problems and creates a CSV
with columns: id, problem_description, correct_answer, and reasoning.
The reasoning column contains the step-by-step thought process for solving the problem.
"""

import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.dataset.grpo import get_reasoning_for_answer
from src.utils.arithmetics import (
    ArithmeticProblemDescriptionGenerator,
    ArithmeticProblemGenerator,
    Mode,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_single_problem_with_reasoning(
    problem_generator: ArithmeticProblemGenerator,
    description_generator: ArithmeticProblemDescriptionGenerator,
    problem_id: int,
) -> dict[str, Any] | None:
    """
    Generate a single problem with reasoning.

    Args:
        problem_generator: The arithmetic problem generator
        description_generator: The problem description generator
        problem_id: The ID for this problem

    Returns:
        Optional[Dict[str, Any]]: Training data entry with reasoning, or None if generation failed
    """
    # Generate a problem
    problem = problem_generator.generate_problem()

    if problem is None:
        return None

    # Generate description
    problem_description, _ = description_generator.generate_description(problem)
    correct_answer = problem.expression
    # Generate reasoning using OpenAI
    logger.info(f"Generating reasoning for problem {problem_id}...")
    reasoning = get_reasoning_for_answer(
        problem_description,
        problem.expression
    )


    # Create training data entry
    training_entry = {
        "id": problem_id,
        "problem_description": problem_description,
        "correct_answer": correct_answer,
        "reasoning": reasoning,
    }

    return training_entry


def generate_training_data(
    num_problems: int, max_workers: int = 8
) -> list[dict[str, Any]]:
    """
    Generate training data with arithmetic problems and reasoning using threading.

    Args:
        num_problems: Number of problems to generate
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing training data with reasoning
    """
    problem_generator = ArithmeticProblemGenerator(mode=Mode.MUL_DIV)
    description_generator = ArithmeticProblemDescriptionGenerator()

    training_data = []
    max_total_attempts = num_problems * 10  # Allow more attempts than problems

    logger.info(
        f"Starting generation of {num_problems} problems with reasoning using {max_workers} workers..."
    )

    # Generate problems in batches to avoid over-submission
    attempts = 0
    while len(training_data) < num_problems and attempts < max_total_attempts:
        # Calculate how many more problems we need
        remaining_problems = num_problems - len(training_data)

        # Submit a batch of tasks (no more than we need + small buffer for failures)
        batch_size = min(
            max_workers, remaining_problems + 2, max_total_attempts - attempts
        )

        logger.info(
            f"Submitting batch of {batch_size} tasks. Need {remaining_problems} more problems."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch of tasks
            futures = []
            for i in range(batch_size):
                future = executor.submit(
                    generate_single_problem_with_reasoning,
                    problem_generator,
                    description_generator,
                    attempts + i + 1,
                )
                futures.append(future)

            # Collect results from this batch
            batch_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)

                        # Stop collecting if we have enough problems
                        if len(training_data) + len(batch_results) >= num_problems:
                            break

                except Exception as e:
                    logger.error(f"Error generating problem: {e}")

            # Add successful results to training data
            for _, result in enumerate(batch_results):
                if len(training_data) >= num_problems:
                    break
                result["id"] = len(training_data) + 1
                training_data.append(result)

                if len(training_data) % 10 == 0:
                    logger.info(
                        f"Generated {len(training_data)} problems with reasoning..."
                    )

        attempts += batch_size

        # Log progress
        logger.info(
            f"Completed batch. Have {len(training_data)} problems, need {num_problems}"
        )

        # Stop if we have enough problems
        if len(training_data) >= num_problems:
            break

    if len(training_data) < num_problems:
        logger.warning(
            f"Only generated {len(training_data)} out of {num_problems} requested problems after {attempts} attempts"
        )
    else:
        logger.info(
            f"Successfully generated {len(training_data)} problems with reasoning in {attempts} attempts"
        )

    # Sort training data by ID to maintain order
    training_data.sort(key=lambda x: x["id"])

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
        fieldnames = ["id", "problem_description", "correct_answer", "reasoning"]
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
        description="Generate arithmetic problems for SFT training data in CSV format with reasoning"
    )

    parser.add_argument(
        "--num_problems", type=int, required=True, help="Number of problems to generate"
    )

    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output CSV file"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of worker threads for parallel processing (default: 8)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_problems <= 0:
        logger.error("Number of problems must be positive")
        return

    if args.max_workers <= 0:
        logger.error("Number of workers must be positive")
        return

    output_path = Path(args.output_file)

    # Generate training data
    training_data = generate_training_data(args.num_problems, args.max_workers)

    if not training_data:
        logger.error("Failed to generate any training data")
        return

    # Save to CSV
    save_to_csv(training_data, output_path)


if __name__ == "__main__":
    main()
