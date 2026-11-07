"""Download and format the UCI HAR dataset for this assignment.

The UCI dataset already provides separate training and test partitions. Keeping
that split intact makes the evaluation honest and supplies the exact test CSV
that the Streamlit app will accept later.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_PATH = RAW_DIR / "uci_har_dataset.zip"
EXTRACTED_DATASET = RAW_DIR / "UCI HAR Dataset"

# The original UCI machine-learning repository hosts this archive publicly.
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)
TARGET_COLUMN = "activity"


def download_and_extract_dataset() -> Path:
    """Return the extracted UCI dataset directory, downloading only when needed."""
    if EXTRACTED_DATASET.exists():
        return EXTRACTED_DATASET

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists():
        print("Downloading the UCI HAR dataset...")
        urllib.request.urlretrieve(UCI_DATASET_URL, ARCHIVE_PATH)
    else:
        print("Using the existing UCI HAR archive.")

    print("Extracting the archive...")
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(RAW_DIR)

    if not EXTRACTED_DATASET.exists():
        raise FileNotFoundError("The expected 'UCI HAR Dataset' folder was not found.")
    return EXTRACTED_DATASET


def read_feature_names(dataset_directory: Path) -> list[str]:
    """Read the 561 sensor features and make every CSV header unambiguous."""
    feature_table = pd.read_csv(
        dataset_directory / "features.txt",
        sep=r"\s+",
        header=None,
        names=["index", "feature_name"],
    )
    # A few descriptive names occur more than once in the UCI source. Prefixing
    # each name with its documented feature number preserves the source label
    # while guaranteeing that pandas and the uploaded CSV have unique columns.
    return [
        f"feature_{feature_index:03d}_{feature_name}"
        for feature_index, feature_name in feature_table.itertuples(index=False)
    ]


def read_activity_labels(dataset_directory: Path) -> dict[int, str]:
    """Map UCI's numeric activity codes to readable activity labels."""
    label_table = pd.read_csv(
        dataset_directory / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["code", TARGET_COLUMN],
    )
    return dict(zip(label_table["code"], label_table[TARGET_COLUMN]))


def build_partition(
    dataset_directory: Path,
    partition_name: str,
    feature_names: list[str],
    activity_labels: dict[int, str],
) -> pd.DataFrame:
    """Create one labelled DataFrame from the original UCI files."""
    partition_directory = dataset_directory / partition_name
    feature_values = pd.read_csv(
        partition_directory / f"X_{partition_name}.txt",
        sep=r"\s+",
        header=None,
        names=feature_names,
    )
    activity_codes = pd.read_csv(
        partition_directory / f"y_{partition_name}.txt",
        header=None,
        names=["activity_code"],
    )

    if len(feature_values) != len(activity_codes):
        raise ValueError(f"The {partition_name} features and labels have different row counts.")

    labelled_partition = feature_values.copy()
    labelled_partition[TARGET_COLUMN] = activity_codes["activity_code"].map(activity_labels)
    if labelled_partition[TARGET_COLUMN].isna().any():
        raise ValueError("One or more activity codes could not be mapped to labels.")
    return labelled_partition


def write_metadata(
    feature_names: list[str], train_data: pd.DataFrame, test_data: pd.DataFrame
) -> None:
    """Write the values the training script and Streamlit app need to agree on."""
    metadata = {
        "dataset_name": "UCI Human Activity Recognition Using Smartphones",
        "dataset_source": UCI_DATASET_URL,
        "target_column": TARGET_COLUMN,
        "feature_names": feature_names,
        "number_of_features": len(feature_names),
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "class_names": sorted(test_data[TARGET_COLUMN].unique().tolist()),
        "split_description": "Official UCI train/test partition retained without reshuffling.",
    }
    with (DATA_DIR / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main() -> None:
    """Create CSV files and validate the assignment's dataset-size requirements."""
    DATA_DIR.mkdir(exist_ok=True)
    dataset_directory = download_and_extract_dataset()
    feature_names = read_feature_names(dataset_directory)
    activity_labels = read_activity_labels(dataset_directory)

    train_data = build_partition(
        dataset_directory, "train", feature_names, activity_labels
    )
    test_data = build_partition(dataset_directory, "test", feature_names, activity_labels)

    if len(feature_names) < 12 or len(train_data) + len(test_data) < 500:
        raise ValueError("The selected dataset does not meet the assignment minimum size.")

    train_data.to_csv(DATA_DIR / "train_data.csv", index=False)
    test_data.to_csv(DATA_DIR / "test_data.csv", index=False)
    write_metadata(feature_names, train_data, test_data)

    print(f"Training rows: {len(train_data):,}")
    print(f"Test rows: {len(test_data):,}")
    print(f"Features: {len(feature_names):,}")
    print(f"Created: {DATA_DIR / 'test_data.csv'}")


if __name__ == "__main__":
    main()
