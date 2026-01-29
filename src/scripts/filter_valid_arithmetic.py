#!/usr/bin/env python3
"""
Script to filter CSV rows based on valid arithmetic expressions.

This script reads a CSV file, extracts expressions from <answer></answer> tags
in the reasoning column, validates them using check_valid_arithmetic_expression,
and outputs only the rows with valid expressions.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.utils.arithmetics import check_valid_arithmetic_expression

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_answer_from_reasoning(reasoning: str) -> str | None:
    """
    Extract the expression from <answer></answer> tags in reasoning text.

    Args:
        reasoning: The reasoning text that may contain <answer></answer> tags

    Returns:
        Optional[str]: The extracted expression or None if not found
    """
    # Use regex to find content between <answer> and </answer> tags
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, reasoning, re.IGNORECASE | re.DOTALL)

    if match:
        # Extract and clean the expression
        expression = match.group(1).strip()
        return expression

    return None


def is_valid_row(reasoning: str, correct_answer: str) -> bool:
    """
    Check if a CSV row contains a valid arithmetic expression.

    Args:
        reasoning: The reasoning column content
        correct_answer: The correct answer column content

    Returns:
        bool: True if the row contains a valid arithmetic expression
    """
    # Extract expression from reasoning
    expression = extract_answer_from_reasoning(reasoning)
    if not expression:
        logger.debug("No expression found in reasoning")
        return False

    # Convert correct_answer to integer
    try:
        result = int(correct_answer)
    except (ValueError, TypeError):
        logger.debug(f"Invalid correct_answer: {correct_answer}")
        return False

    # Validate the arithmetic expression
    is_valid = check_valid_arithmetic_expression(expression, result)
    if is_valid:
        logger.debug(f"Valid expression: {expression} = {result}")
    else:
        logger.debug(f"Invalid expression: {expression} != {result}")

    return is_valid


def filter_csv_file(
    input_file: Path,
    output_file: Path,
    reasoning_col: str = "reasoning",
    correct_answer_col: str = "correct_answer",
) -> None:
    """
    Filter CSV file to keep only rows with valid arithmetic expressions.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        reasoning_col: Name of the reasoning column
        correct_answer_col: Name of the correct answer column
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read CSV file using pandas
    df = pd.read_csv(input_file)

    # Check if required columns exist
    if reasoning_col not in df.columns:
        raise ValueError(f"Column '{reasoning_col}' not found in CSV file")
    if correct_answer_col not in df.columns:
        raise ValueError(f"Column '{correct_answer_col}' not found in CSV file")

    total_rows = len(df)

    # Apply validation to each row
    valid_mask = df.apply(
        lambda row: is_valid_row(row[reasoning_col], row[correct_answer_col]), axis=1
    )

    # Filter the dataframe to keep only valid rows
    filtered_df = df[valid_mask]
    valid_rows = len(filtered_df)

    # Save filtered dataframe to output file
    filtered_df.to_csv(output_file, index=False)

    logger.info(f"Processed {total_rows} rows, kept {valid_rows} valid rows")
    logger.info(f"Filtered CSV saved to: {output_file}")


def main() -> None:
    """Main function to handle command line arguments and run the filtering."""
    parser = argparse.ArgumentParser(
        description="Filter CSV rows based on valid arithmetic expressions in <answer></answer> tags"
    )
    parser.add_argument(
        "--input_file", type=Path, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="Path to output CSV file"
    )

    args = parser.parse_args()

    try:
        filter_csv_file(args.input_file, args.output_file)
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise


if __name__ == "__main__":
    main()
