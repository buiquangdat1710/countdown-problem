#!/usr/bin/env python3
"""
Script to split a dataset into train and validation sets.

This script takes a CSV dataset and splits it into training and validation sets
based on specified ratios. It supports stratified splitting, random seed control
for reproducibility, and maintains data integrity across splits.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_input_file(file_path: Path) -> None:
    """
    Validate the input CSV file exists and is readable.

    Args:
        file_path: Path to the input CSV file

    Raises:
        ValueError: If file validation fails
    """
    if not file_path.exists():
        raise ValueError(f"Input file does not exist: {file_path}")

    if not file_path.suffix.lower() == ".csv":
        raise ValueError(f"Input file is not a CSV file: {file_path}")

    try:
        # Try to read the file to ensure it's valid
        pd.read_csv(file_path, nrows=1)
        logger.info(f"Input file validated: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}") from e


def validate_split_parameters(
    train_ratio: float, val_ratio: float, test_ratio: float | None = None
) -> None:
    """
    Validate split ratio parameters.

    Args:
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio (optional)

    Raises:
        ValueError: If split ratios are invalid
    """
    ratios = [train_ratio, val_ratio]
    if test_ratio is not None:
        ratios.append(test_ratio)

    # Check individual ratios
    for ratio in ratios:
        if not 0 < ratio < 1:
            raise ValueError(f"Split ratio must be between 0 and 1, got: {ratio}")

    # Check sum of ratios
    total_ratio = sum(ratios)
    if not 0.99 <= total_ratio <= 1.01:  # Allow small floating point errors
        raise ValueError(
            f"Split ratios must sum to 1.0, got: {total_ratio:.3f} "
            f"(train: {train_ratio}, val: {val_ratio}"
            + (f", test: {test_ratio}" if test_ratio else "")
            + ")"
        )


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        ValueError: If dataset loading fails
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {file_path}: {e}") from e


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None = None,
    stratify_column: str | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Split dataset into train, validation, and optionally test sets.

    Args:
        df: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio (optional)
        stratify_column: Column name for stratified splitting (optional)
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df) where test_df is None if test_ratio is None

    Raises:
        ValueError: If splitting fails
    """
    # Validate stratify column if provided
    stratify_data = None
    if stratify_column:
        if stratify_column not in df.columns:
            raise ValueError(
                f"Stratify column '{stratify_column}' not found in dataset"
            )
        stratify_data = df[stratify_column]
        logger.info(f"Using stratified splitting based on column: {stratify_column}")

    try:
        if test_ratio is None:
            # Simple train-validation split
            train_df, val_df = train_test_split(
                df,
                train_size=train_ratio,
                test_size=val_ratio,
                random_state=random_seed,
                stratify=stratify_data,
                shuffle=True,
            )
            test_df = None

            logger.info("Dataset split completed:")
            logger.info(
                f"  Training set: {len(train_df)} rows ({len(train_df) / len(df):.1%})"
            )
            logger.info(
                f"  Validation set: {len(val_df)} rows ({len(val_df) / len(df):.1%})"
            )

        else:
            # Three-way split: train-validation-test
            # First split into train and temp (val+test)
            temp_ratio = val_ratio + test_ratio
            train_df, temp_df = train_test_split(
                df,
                train_size=train_ratio,
                test_size=temp_ratio,
                random_state=random_seed,
                stratify=stratify_data,
                shuffle=True,
            )

            # Then split temp into validation and test
            # Calculate relative ratios for the second split
            val_relative_ratio = val_ratio / temp_ratio
            test_relative_ratio = test_ratio / temp_ratio

            # Update stratify data for second split if needed
            temp_stratify = None
            if stratify_data is not None:
                temp_stratify = temp_df[stratify_column]

            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_relative_ratio,
                test_size=test_relative_ratio,
                random_state=random_seed + 1,  # Different seed for second split
                stratify=temp_stratify,
                shuffle=True,
            )

            logger.info("Dataset split completed:")
            logger.info(
                f"  Training set: {len(train_df)} rows ({len(train_df) / len(df):.1%})"
            )
            logger.info(
                f"  Validation set: {len(val_df)} rows ({len(val_df) / len(df):.1%})"
            )
            logger.info(
                f"  Test set: {len(test_df)} rows ({len(test_df) / len(df):.1%})"
            )

        return train_df, val_df, test_df

    except Exception as e:
        raise ValueError(f"Failed to split dataset: {e}") from e


def save_split_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    output_dir: Path,
    base_name: str,
) -> None:
    """
    Save split datasets to CSV files.

    Args:
        train_df: Training dataset
        val_df: Validation dataset
        test_df: Test dataset (optional)
        output_dir: Output directory
        base_name: Base name for output files

    Raises:
        ValueError: If saving fails
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        train_path = output_dir / f"{base_name}_train.csv"
        val_path = output_dir / f"{base_name}_val.csv"

        # Save train and validation sets
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        logger.info(f"Saved training set to: {train_path}")
        logger.info(f"Saved validation set to: {val_path}")

        # Save test set if provided
        if test_df is not None:
            test_path = output_dir / f"{base_name}_test.csv"
            test_df.to_csv(test_path, index=False)
            logger.info(f"Saved test set to: {test_path}")

    except Exception as e:
        raise ValueError(f"Failed to save split datasets: {e}") from e


def analyze_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    stratify_column: str | None = None,
) -> None:
    """
    Analyze and log the distribution of data across splits.

    Args:
        train_df: Training dataset
        val_df: Validation dataset
        test_df: Test dataset (optional)
        stratify_column: Column used for stratification (optional)
    """
    logger.info("Dataset split analysis:")

    # Basic statistics
    total_rows = (
        len(train_df) + len(val_df) + (len(test_df) if test_df is not None else 0)
    )
    logger.info(f"  Total rows: {total_rows}")

    # If stratify column is provided, analyze distribution
    if stratify_column and stratify_column in train_df.columns:
        logger.info(f"Distribution analysis for '{stratify_column}':")

        train_dist = train_df[stratify_column].value_counts().sort_index()
        val_dist = val_df[stratify_column].value_counts().sort_index()

        logger.info("  Training set distribution:")
        for value, count in train_dist.items():
            percentage = count / len(train_df) * 100
            logger.info(f"    {value}: {count} ({percentage:.1f}%)")

        logger.info("  Validation set distribution:")
        for value, count in val_dist.items():
            percentage = count / len(val_df) * 100
            logger.info(f"    {value}: {count} ({percentage:.1f}%)")

        if test_df is not None:
            test_dist = test_df[stratify_column].value_counts().sort_index()
            logger.info("  Test set distribution:")
            for value, count in test_dist.items():
                percentage = count / len(test_df) * 100
                logger.info(f"    {value}: {count} ({percentage:.1f}%)")


def main() -> None:
    """
    Main function to split dataset into train and validation sets.
    """
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 80-20 train-validation split
  python split_dataset.py --input_path data/dataset.csv --output_dir data/splits --train_ratio 0.8 --val_ratio 0.2

  # Three-way split with test set
  python split_dataset.py --input_path data/dataset.csv --output_dir data/splits --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

  # Stratified split based on a column
  python split_dataset.py --input_path data/dataset.csv --output_dir data/splits --train_ratio 0.8 --val_ratio 0.2 --stratify_column target_result

  # Custom base name and random seed
  python split_dataset.py --input_path data/dataset.csv --output_dir data/splits --train_ratio 0.8 --val_ratio 0.2 --base_name countdown --random_seed 123
        """,
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV file to split",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save split datasets",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        required=True,
        help="Ratio for training set (e.g., 0.8 for 80%)",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
        help="Ratio for validation set (e.g., 0.2 for 20%)",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        help="Ratio for test set (optional, e.g., 0.1 for 10%)",
    )

    parser.add_argument(
        "--stratify_column",
        type=str,
        help="Column name to use for stratified splitting (optional)",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )

    parser.add_argument(
        "--base_name",
        type=str,
        default="dataset",
        help="Base name for output files (default: 'dataset')",
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
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    try:
        logger.info("Starting dataset splitting process")
        logger.info(f"Input file: {input_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Train ratio: {args.train_ratio}")
        logger.info(f"Validation ratio: {args.val_ratio}")
        if args.test_ratio:
            logger.info(f"Test ratio: {args.test_ratio}")
        if args.stratify_column:
            logger.info(f"Stratify column: {args.stratify_column}")
        logger.info(f"Random seed: {args.random_seed}")

        # Validate input file
        validate_input_file(input_path)

        # Validate split parameters
        validate_split_parameters(args.train_ratio, args.val_ratio, args.test_ratio)

        # Load dataset
        df = load_dataset(input_path)

        # Split dataset
        train_df, val_df, test_df = split_dataset(
            df=df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_column=args.stratify_column,
            random_seed=args.random_seed,
        )

        # Save split datasets
        save_split_datasets(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=output_dir,
            base_name=args.base_name,
        )

        # Analyze split distribution
        analyze_split_distribution(train_df, val_df, test_df, args.stratify_column)

        logger.info("Dataset splitting completed successfully")

    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        raise


if __name__ == "__main__":
    main()
