#!/usr/bin/env python3
"""
Script to combine multiple CSV datasets into a single dataset file.

This script takes multiple input CSV files and concatenates them into a single
output CSV file. It handles deduplication, ID reassignment, and validates
data consistency across input files.
"""

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_csv_structure(
    file_path: Path, expected_columns: set[str] | None = None
) -> set[str]:
    """
    Validate the structure of a CSV file and return its columns.

    Args:
        file_path: Path to the CSV file to validate
        expected_columns: Optional set of expected column names for validation

    Returns:
        set[str]: Set of column names found in the file

    Raises:
        ValueError: If file validation fails
    """
    if not file_path.exists():
        raise ValueError(f"Input file does not exist: {file_path}")

    if not file_path.suffix.lower() == ".csv":
        raise ValueError(f"Input file is not a CSV file: {file_path}")

    try:
        # Read just the header to validate structure
        df_sample = pd.read_csv(file_path, nrows=0)
        columns = set(df_sample.columns)

        if expected_columns and columns != expected_columns:
            raise ValueError(
                f"Column mismatch in {file_path}. "
                f"Expected: {expected_columns}, Found: {columns}"
            )

        logger.info(f"Validated {file_path}: {len(columns)} columns found")
        return columns

    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}") from e


def load_and_combine_datasets(
    input_paths: list[Path],
    remove_duplicates: bool = True,
    duplicate_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and combine multiple CSV datasets into a single DataFrame.

    Args:
        input_paths: List of paths to input CSV files
        remove_duplicates: Whether to remove duplicate rows
        duplicate_columns: Columns to use for duplicate detection (if None, uses all columns)

    Returns:
        pd.DataFrame: Combined dataset with reassigned IDs

    Raises:
        ValueError: If datasets cannot be combined
    """
    if not input_paths:
        raise ValueError("No input paths provided")

    dataframes = []
    expected_columns = None

    for i, path in enumerate(input_paths):
        logger.info(f"Loading dataset {i + 1}/{len(input_paths)}: {path}")

        # Validate file structure
        columns = validate_csv_structure(path, expected_columns)
        if expected_columns is None:
            expected_columns = columns

        # Load the dataset
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows from {path}")
            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {path}: {e}") from e

    # Combine all dataframes
    logger.info("Combining datasets...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset has {len(combined_df)} total rows")

    # Remove duplicates if requested
    if remove_duplicates:
        initial_count = len(combined_df)
        if duplicate_columns:
            combined_df = combined_df.drop_duplicates(
                subset=duplicate_columns, keep="first"
            )
        else:
            combined_df = combined_df.drop_duplicates(keep="first")

        duplicates_removed = initial_count - len(combined_df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

    # Reassign IDs if 'id' column exists
    if "id" in combined_df.columns:
        logger.info("Reassigning IDs sequentially...")
        combined_df["id"] = range(1, len(combined_df) + 1)

    return combined_df


def save_combined_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the combined dataset to a CSV file.

    Args:
        df: DataFrame to save
        output_path: Path where to save the combined dataset

    Raises:
        ValueError: If save operation fails
    """
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Successfully saved combined dataset to {output_path}")
        logger.info(
            f"Final dataset contains {len(df)} rows and {len(df.columns)} columns"
        )

        # Log column names for verification
        logger.info(f"Columns: {', '.join(df.columns)}")

    except Exception as e:
        raise ValueError(
            f"Failed to save combined dataset to {output_path}: {e}"
        ) from e


def main() -> None:
    """
    Main function to combine multiple datasets.
    """
    parser = argparse.ArgumentParser(
        description="Combine multiple CSV datasets into a single dataset file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine two datasets
  python combine_dataset.py --input_paths data/train1.csv data/train2.csv --output_path data/combined_train.csv

  # Combine multiple datasets without removing duplicates
  python combine_dataset.py --input_paths data/*.csv --output_path data/all_data.csv --no-remove-duplicates

  # Combine datasets with custom duplicate detection columns
  python combine_dataset.py --input_paths data/train*.csv --output_path data/combined.csv --duplicate_columns problem_description correct_answer
        """,
    )

    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input CSV files to combine",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for the output combined CSV file",
    )

    parser.add_argument(
        "--no-remove-duplicates",
        action="store_true",
        help="Do not remove duplicate rows from the combined dataset",
    )

    parser.add_argument(
        "--duplicate_columns",
        type=str,
        nargs="*",
        help="Columns to use for duplicate detection (if not specified, uses all columns)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Convert string paths to Path objects
    input_paths = [Path(path) for path in args.input_paths]
    output_path = Path(args.output_path)

    try:
        logger.info(f"Starting dataset combination with {len(input_paths)} input files")
        logger.info(f"Input files: {[str(p) for p in input_paths]}")
        logger.info(f"Output file: {output_path}")

        # Load and combine datasets
        combined_df = load_and_combine_datasets(
            input_paths=input_paths,
            remove_duplicates=not args.no_remove_duplicates,
            duplicate_columns=args.duplicate_columns,
        )

        # Save combined dataset
        save_combined_dataset(combined_df, output_path)

        logger.info("Dataset combination completed successfully")

    except Exception as e:
        logger.error(f"Dataset combination failed: {e}")
        raise


if __name__ == "__main__":
    main()
