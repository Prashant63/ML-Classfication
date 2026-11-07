"""Evaluate the saved classifiers on the untouched UCI test partition."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

try:
    # Used when imported as src.evaluate_models by the Jupyter notebook.
    from .evaluation_utils import (
        MODEL_DISPLAY_NAMES,
        calculate_evaluation,
        create_classification_report,
        create_confusion_matrix,
    )
except ImportError:
    # Used when run directly with: python src/evaluate_models.py
    from evaluation_utils import (
        MODEL_DISPLAY_NAMES,
        calculate_evaluation,
        create_classification_report,
        create_confusion_matrix,
    )


# This script writes figures to files, so it does not need a desktop display.
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_metadata() -> dict:
    """Read the training contract shared by every saved model."""
    with (MODEL_DIR / "model_metadata.json").open(encoding="utf-8") as source:
        return json.load(source)


def main() -> None:
    """Write the assignment comparison table, reports, and confusion matrices."""
    RESULTS_DIR.mkdir(exist_ok=True)
    metadata = load_metadata()
    test_data = pd.read_csv(DATA_DIR / "test_data.csv")
    metric_rows = []
    classification_reports = {}

    for model_name, model_filename in metadata["model_files"].items():
        print(f"Evaluating {model_name}...")
        model = joblib.load(MODEL_DIR / model_filename)
        metrics, actual_labels, predicted_labels = calculate_evaluation(
            model, test_data, metadata
        )
        metric_rows.append({"ML Model Name": MODEL_DISPLAY_NAMES[model_name], **metrics})

        report = create_classification_report(
            actual_labels, predicted_labels, metadata["class_names"]
        )
        classification_reports[MODEL_DISPLAY_NAMES[model_name]] = report.to_dict()

        figure = create_confusion_matrix(
            actual_labels, predicted_labels, metadata["class_names"]
        )
        figure.savefig(RESULTS_DIR / f"{model_name}_confusion_matrix.png", dpi=180)
        figure.clear()

    comparison_table = pd.DataFrame(metric_rows)
    comparison_table.to_csv(RESULTS_DIR / "metrics.csv", index=False, float_format="%.4f")
    with (RESULTS_DIR / "classification_reports.json").open("w", encoding="utf-8") as output:
        json.dump(classification_reports, output, indent=2)

    print("\nEvaluation metrics")
    print(comparison_table.round(4).to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
