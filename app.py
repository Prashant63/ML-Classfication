"""Interactive Streamlit interface for evaluating saved activity classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.evaluation_utils import (
    MODEL_DISPLAY_NAMES,
    calculate_evaluation,
    create_classification_report,
    create_confusion_matrix,
    validate_test_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_data
def load_model_metadata() -> dict:
    """Read the feature contract shared by all saved classifiers."""
    with (MODEL_DIR / "model_metadata.json").open(encoding="utf-8") as source:
        return json.load(source)


@st.cache_resource
def load_saved_model(model_name: str):
    """Load one fitted model once; this function never trains a model."""
    metadata = load_model_metadata()
    return joblib.load(MODEL_DIR / metadata["model_files"][model_name])


@st.cache_data
def load_included_test_data() -> pd.DataFrame:
    """Load the included UCI test data for an immediate reproducible demo."""
    return pd.read_csv(DATA_DIR / "test_data.csv")


@st.cache_data
def load_saved_comparison() -> pd.DataFrame:
    """Load the all-model comparison calculated during the offline evaluation."""
    return pd.read_csv(RESULTS_DIR / "metrics.csv")


def display_metrics(metrics: dict[str, float]) -> None:
    """Present the six required evaluation metrics in a compact layout."""
    first_row = st.columns(3)
    second_row = st.columns(3)
    metric_locations = first_row + second_row
    for location, (metric_name, value) in zip(metric_locations, metrics.items()):
        location.metric(metric_name, f"{value:.4f}")


def get_test_data() -> tuple[pd.DataFrame, str]:
    """Use an uploaded CSV when supplied; otherwise use the included test CSV."""
    uploaded_file = st.file_uploader(
        "Upload labelled test data (CSV)",
        type="csv",
        help=(
            "The file must include the 561 required feature columns and the "
            "'activity' target column. The app evaluates a saved model only."
        ),
    )
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "uploaded test data"
    return load_included_test_data(), "included UCI test data"


def main() -> None:
    """Run the assignment interface."""
    st.set_page_config(page_title="Activity Classification Lab", page_icon="🧠", layout="wide")
    st.title("Activity Classification Lab")
    st.caption(
        "A saved-model evaluation dashboard for the UCI Human Activity Recognition dataset"
    )
    st.info(
        "This app never retrains a model. It loads a saved classifier, evaluates it "
        "on labelled test data, and shows the required metrics and diagnostics."
    )

    metadata = load_model_metadata()
    display_to_key = {display_name: key for key, display_name in MODEL_DISPLAY_NAMES.items()}
    selected_display_name = st.selectbox("Choose a classification model", list(display_to_key))
    selected_model_name = display_to_key[selected_display_name]

    try:
        test_data, data_source = get_test_data()
        validate_test_data(test_data, metadata)
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"The test CSV cannot be evaluated: {error}")
        st.stop()

    st.subheader(f"{selected_display_name} results")
    st.caption(f"Evaluating {len(test_data):,} rows from the {data_source}.")

    model = load_saved_model(selected_model_name)
    metrics, actual_labels, predicted_labels = calculate_evaluation(
        model, test_data, metadata
    )
    display_metrics(metrics)

    report_column, matrix_column = st.columns(2)
    with report_column:
        st.markdown("#### Classification report")
        report = create_classification_report(
            actual_labels, predicted_labels, metadata["class_names"]
        )
        st.dataframe(report.round(4), width="stretch")
    with matrix_column:
        st.markdown("#### Confusion matrix")
        figure = create_confusion_matrix(
            actual_labels, predicted_labels, metadata["class_names"]
        )
        st.pyplot(figure, clear_figure=True, width="stretch")

    st.divider()
    st.subheader("All five models on the included UCI test data")
    st.caption(
        "This comparison is generated offline from the same untouched UCI test partition. "
        "Select another model above to inspect its detailed report and confusion matrix."
    )
    st.dataframe(load_saved_comparison(), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
