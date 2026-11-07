"""Train and save the five classifiers required by the assignment."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RANDOM_STATE = 42


def load_training_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load features, labels, and the shared feature contract from disk."""
    with (DATA_DIR / "metadata.json").open(encoding="utf-8") as source:
        metadata = json.load(source)

    training_data = pd.read_csv(DATA_DIR / "train_data.csv")
    feature_names = metadata["feature_names"]
    target_column = metadata["target_column"]

    expected_columns = set(feature_names + [target_column])
    if set(training_data.columns) != expected_columns:
        raise ValueError("Training CSV columns do not match the saved feature metadata.")

    return training_data[feature_names], training_data[target_column], metadata


def build_models() -> dict[str, Pipeline | DecisionTreeClassifier | GaussianNB]:
    """Create simple, documented baseline configurations for all five models."""
    return {
        "logistic_regression": Pipeline(
            [
                # Distance and gradient-based models are sensitive to feature scale.
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=20, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "gaussian_naive_bayes": GaussianNB(),
        "support_vector_machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=2.0,
                        kernel="rbf",
                        gamma="scale",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def save_model_metadata(metadata: dict, model_names: list[str]) -> None:
    """Save a small contract so the Streamlit app loads each artifact correctly."""
    model_metadata = {
        "dataset_name": metadata["dataset_name"],
        "target_column": metadata["target_column"],
        "feature_names": metadata["feature_names"],
        "class_names": metadata["class_names"],
        "training_rows": metadata["training_rows"],
        "model_files": {name: f"{name}.joblib" for name in model_names},
        "random_state": RANDOM_STATE,
    }
    with (MODEL_DIR / "model_metadata.json").open("w", encoding="utf-8") as output:
        json.dump(model_metadata, output, indent=2)


def main() -> None:
    """Fit every model on the same UCI training partition and persist it."""
    MODEL_DIR.mkdir(exist_ok=True)
    features, labels, metadata = load_training_data()
    models = build_models()

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(features, labels)
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib", compress=3)

    save_model_metadata(metadata, list(models))
    print(f"Saved {len(models)} model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
