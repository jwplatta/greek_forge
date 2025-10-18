"""Tests for evaluator module."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.evaluator import (
    evaluate_model,
    generate_evaluation_report,
    analyze_predictions,
)


class TestEvaluateModel:
    """Test evaluate_model function."""

    def test_evaluate_model_returns_dict(self):
        """Test that evaluate_model returns a dictionary."""
        # Create simple test data
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        y_train = pd.Series([1, 2, 3, 4, 5])
        X_test = pd.DataFrame({"feature1": [6, 7, 8]})
        y_test = pd.Series([6, 7, 8])

        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        result = evaluate_model(model, X_test, y_test)

        assert isinstance(result, dict)
        assert "test" in result

    def test_evaluate_model_includes_train_metrics(self):
        """Test that train metrics are included when provided."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        y_train = pd.Series([1, 2, 3, 4, 5])
        X_test = pd.DataFrame({"feature1": [6, 7, 8]})
        y_test = pd.Series([6, 7, 8])

        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        result = evaluate_model(model, X_test, y_test, X_train, y_train)

        assert "test" in result
        assert "train" in result


class TestAnalyzePredictions:
    """Test analyze_predictions function."""

    def test_analyze_predictions_returns_dict(self):
        """Test that analyze_predictions returns a dictionary."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        result = analyze_predictions(y_true, y_pred)

        assert isinstance(result, dict)
        assert "mean_error" in result
        assert "error_bins" in result


class TestGenerateEvaluationReport:
    """Test generate_evaluation_report function."""

    def test_print_to_console_no_error(self):
        """Test that printing to console doesn't raise an error."""
        metrics = {
            "test": {"mae": 0.1, "rmse": 0.15, "r2": 0.95},
            "train": {"mae": 0.08, "rmse": 0.12, "r2": 0.97},
        }

        # Should not raise an error
        generate_evaluation_report(metrics)

    def test_write_to_file_creates_file(self):
        """Test that writing to file creates the file."""
        metrics = {
            "test": {"mae": 0.1, "rmse": 0.15, "r2": 0.95},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.txt"
            generate_evaluation_report(metrics, output_file=output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_write_to_file_contains_metrics(self):
        """Test that written file contains the metrics."""
        metrics = {
            "test": {"mae": 0.123456, "rmse": 0.234567, "r2": 0.945678},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.txt"
            generate_evaluation_report(metrics, output_file=output_path)

            content = output_path.read_text()
            assert "MODEL EVALUATION REPORT" in content
            assert "Test Set Performance" in content
            assert "0.123456" in content  # MAE
            assert "0.234567" in content  # RMSE
            assert "0.945678" in content  # R²

    def test_write_creates_parent_directories(self):
        """Test that parent directories are created if they don't exist."""
        metrics = {
            "test": {"mae": 0.1, "rmse": 0.15, "r2": 0.95},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "report.txt"
            generate_evaluation_report(metrics, output_file=output_path)

            assert output_path.exists()
            assert output_path.parent.exists()
