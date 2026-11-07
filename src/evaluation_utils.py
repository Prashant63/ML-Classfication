"""Shared evaluation helpers for the command-line script and Streamlit app."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_naive_bayes": "Gaussian Naive Bayes",
    "support_vector_machine": "Support Vector Machine",
}


def validate_test_data(test_data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Ensure an uploaded CSV has all inputs and the expected target column."""
    required_columns = set(metadata["feature_names"] + [metadata["target_column"]])
    missing_columns = sorted(required_columns - set(test_data.columns))
    if missing_columns:
        example_columns = ", ".join(missing_columns[:5])
        raise ValueError(
            f"The uploaded CSV is missing {len(missing_columns)} required columns. "
            f"Examples: {example_columns}"
        )
    if test_data[metadata["feature_names"]].isna().any().any():
        raise ValueError("The uploaded feature columns contain missing values.")
    if test_data[metadata["target_column"]].isna().any():
        raise ValueError("The uploaded target column contains missing values.")


def get_model_scores(model: Any, features: pd.DataFrame) -> tuple[Any, bool]:
    """Return scores for AUC and say whether they are calibrated probabilities."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features), True
    if hasattr(model, "decision_function"):
        return model.decision_function(features), False
    raise TypeError("The saved model cannot provide scores needed for AUC.")


def calculate_multiclass_auc(
    actual_labels: pd.Series,
    model_scores: Any,
    class_names: list[str],
    scores_are_probabilities: bool,
) -> float:
    """Calculate macro one-vs-rest AUC from probabilities or SVM decision scores."""
    if scores_are_probabilities:
        return float(
            roc_auc_score(
                actual_labels,
                model_scores,
                multi_class="ovr",
                average="macro",
                labels=class_names,
            )
        )

    # An SVM's decision function is not a probability distribution, so sklearn's
    # multiclass shortcut rejects it. Instead, calculate each one-vs-rest AUC
    # from the class score and take the macro mean, which is the same AUC design.
    binary_labels = label_binarize(actual_labels, classes=class_names)
    per_class_auc = [
        roc_auc_score(binary_labels[:, class_index], model_scores[:, class_index])
        for class_index in range(len(class_names))
    ]
    return float(np.mean(per_class_auc))


def calculate_evaluation(
    model: Any, test_data: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    """Calculate every required metric from a fitted model and labelled test data."""
    validate_test_data(test_data, metadata)
    features = test_data[metadata["feature_names"]]
    actual_labels = test_data[metadata["target_column"]]
    predicted_labels = pd.Series(model.predict(features), index=test_data.index)
    model_scores, scores_are_probabilities = get_model_scores(model, features)

    # With six activity classes, one-vs-rest AUC evaluates each class against
    # the remaining classes. Macro averaging gives every activity equal weight.
    auc_score = calculate_multiclass_auc(
        actual_labels,
        model_scores,
        list(model.classes_),
        scores_are_probabilities,
    )
    metrics = {
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "AUC": auc_score,
        "Precision": precision_score(
            actual_labels, predicted_labels, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            actual_labels, predicted_labels, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            actual_labels, predicted_labels, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(actual_labels, predicted_labels),
    }
    return metrics, actual_labels, predicted_labels


def create_classification_report(
    actual_labels: pd.Series, predicted_labels: pd.Series, class_names: list[str]
) -> pd.DataFrame:
    """Return a readable per-class precision, recall, and F1 table."""
    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def create_confusion_matrix(
    actual_labels: pd.Series, predicted_labels: pd.Series, class_names: list[str]
):
    """Build a labelled confusion-matrix figure for a notebook, report, or app."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = confusion_matrix(actual_labels, predicted_labels, labels=class_names)
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted activity")
    axis.set_ylabel("Actual activity")
    figure.tight_layout()
    return figure
