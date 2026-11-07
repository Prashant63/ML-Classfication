"""Checks that the saved assignment artifacts agree with one another."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml_assignment_matplotlib")
)


class AssignmentArtifactTests(unittest.TestCase):
    """Verify the data, models, offline results, and Streamlit entry point."""

    @classmethod
    def setUpClass(cls) -> None:
        with (PROJECT_ROOT / "model" / "model_metadata.json").open(encoding="utf-8") as source:
            cls.metadata = json.load(source)
        cls.test_data = pd.read_csv(PROJECT_ROOT / "data" / "test_data.csv", nrows=3)

    def test_dataset_contract(self) -> None:
        """The upload CSV contract has 561 features, a target, and no missing values."""
        feature_names = self.metadata["feature_names"]
        self.assertEqual(len(feature_names), 561)
        self.assertEqual(len(feature_names), len(set(feature_names)))
        self.assertIn(self.metadata["target_column"], self.test_data.columns)
        self.assertFalse(self.test_data[feature_names].isna().any().any())

    def test_saved_models_make_predictions(self) -> None:
        """Every persisted model loads and produces one label for each input row."""
        features = self.test_data[self.metadata["feature_names"]]
        for model_name, model_filename in self.metadata["model_files"].items():
            with self.subTest(model=model_name):
                model = joblib.load(PROJECT_ROOT / "model" / model_filename)
                predictions = model.predict(features)
                self.assertEqual(len(predictions), len(features))
                self.assertTrue(set(predictions).issubset(set(self.metadata["class_names"])))

    def test_results_table_has_all_required_metrics(self) -> None:
        """The offline comparison contains five models and all six required metrics."""
        results = pd.read_csv(PROJECT_ROOT / "results" / "metrics.csv")
        expected_columns = [
            "ML Model Name",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
        ]
        self.assertEqual(results.columns.tolist(), expected_columns)
        self.assertEqual(len(results), 5)
        self.assertTrue(results[expected_columns[1:]].apply(lambda column: column.between(0, 1)).all().all())

    def test_streamlit_app_loads_saved_artifacts(self) -> None:
        """The app exposes the required controls and evaluates without a fit call."""
        from streamlit.testing.v1 import AppTest

        app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", app_source)

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.metric), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
