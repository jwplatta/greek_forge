"""Tests for preprocessor module."""

import pytest
import pandas as pd

from src.data.loader import fetch_call_samples
from src.data.preprocessor import (
    OptionPreprocessor,
    balance_delta_samples,
    preprocess_training_data,
    preprocess_prediction_data,
)


class TestOptionPreprocessor:
    """Test OptionPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes correctly."""
        preprocessor = OptionPreprocessor()

        assert preprocessor.is_fitted is False
        assert len(preprocessor.feature_columns) == 8

    def test_preprocessor_fit(self):
        """Test that preprocessor can be fitted."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        preprocessor.fit(df)

        assert preprocessor.is_fitted is True

    def test_preprocessor_transform_without_fit_raises_error(self):
        """Test that transform without fit raises ValueError."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(df)

    def test_preprocessor_fit_transform(self):
        """Test that fit_transform works."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        transformed_df = preprocessor.fit_transform(df)

        assert isinstance(transformed_df, pd.DataFrame)
        assert preprocessor.is_fitted is True

    def test_preprocessor_selects_correct_features(self):
        """Test that preprocessor selects only expected features."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        transformed_df = preprocessor.fit_transform(df, include_target=False)

        expected_features = [
            "dte",
            "moneyness",
            "mark",
            "strike",
            "underlying_price",
            "vix9d",
            "vvix",
            "skew",
        ]

        assert list(transformed_df.columns) == expected_features

    def test_preprocessor_includes_target_when_requested(self):
        """Test that target is included when include_target=True."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        transformed_df = preprocessor.fit_transform(df, include_target=True)

        assert "delta" in transformed_df.columns

    def test_preprocessor_excludes_target_when_not_requested(self):
        """Test that target is excluded when include_target=False."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        transformed_df = preprocessor.fit_transform(df, include_target=False)

        assert "delta" not in transformed_df.columns


class TestBalanceDeltaSamples:
    """Test delta balancing functionality."""

    def test_balance_delta_samples_returns_dataframe(self):
        """Test that balance_delta_samples returns a DataFrame."""
        df = fetch_call_samples(limit=100)
        balanced_df = balance_delta_samples(df, n_bins=5, samples_per_bin=10)

        assert isinstance(balanced_df, pd.DataFrame)

    def test_balance_delta_samples_size(self):
        """Test that balancing produces expected number of samples."""
        df = fetch_call_samples(limit=100)
        n_bins = 5
        samples_per_bin = 10

        balanced_df = balance_delta_samples(
            df, n_bins=n_bins, samples_per_bin=samples_per_bin
        )

        # Should have n_bins * samples_per_bin rows
        expected_size = n_bins * samples_per_bin
        assert len(balanced_df) == expected_size

    def test_balance_removes_delta_bin_column(self):
        """Test that temporary delta_bin column is removed."""
        df = fetch_call_samples(limit=100)
        balanced_df = balance_delta_samples(df, n_bins=5, samples_per_bin=10)

        assert "delta_bin" not in balanced_df.columns

    def test_balance_uses_random_state(self):
        """Test that random_state makes balancing reproducible."""
        df = fetch_call_samples(limit=100)

        balanced_df1 = balance_delta_samples(
            df, n_bins=5, samples_per_bin=10, random_state=42
        )
        balanced_df2 = balance_delta_samples(
            df, n_bins=5, samples_per_bin=10, random_state=42
        )

        # Results should be identical with same random_state
        pd.testing.assert_frame_equal(balanced_df1, balanced_df2)


class TestPreprocessTrainingData:
    """Test preprocess_training_data function."""

    def test_preprocess_training_data_returns_five_values(self):
        """Test that preprocess_training_data returns 5 values."""
        df = fetch_call_samples(limit=100)
        result = preprocess_training_data(df, test_size=0.3, balance_samples=False)

        assert len(result) == 5

    def test_preprocess_training_data_types(self):
        """Test that returned types are correct."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(
            df, test_size=0.3, balance_samples=False
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(preprocessor, OptionPreprocessor)

    def test_preprocess_training_data_test_size(self):
        """Test that test_size parameter works correctly."""
        df = fetch_call_samples(limit=100)
        test_size = 0.3

        X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(
            df, test_size=test_size, balance_samples=False
        )

        total_samples = len(X_train) + len(X_test)
        actual_test_ratio = len(X_test) / total_samples

        # Allow small tolerance due to rounding
        assert abs(actual_test_ratio - test_size) < 0.05

    def test_preprocess_training_data_no_data_leakage(self):
        """Test that train and test sets don't overlap."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(
            df, test_size=0.3, balance_samples=False
        )

        # Check that indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0

    def test_preprocess_training_data_preprocessor_is_fitted(self):
        """Test that returned preprocessor is fitted."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(
            df, test_size=0.3, balance_samples=False
        )

        assert preprocessor.is_fitted is True

    def test_preprocess_training_data_with_balancing(self):
        """Test that balancing option works in preprocessing pipeline."""
        df = fetch_call_samples()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(
            df, balance_samples=True, n_bins=5, samples_per_bin=100, test_size=0.3
        )

        # Total samples should be n_bins * samples_per_bin
        total_samples = len(X_train) + len(X_test)
        expected_total = 5 * 100
        assert total_samples == expected_total


class TestPreprocessPredictionData:
    """Test preprocess_prediction_data function."""

    def test_preprocess_prediction_data_returns_dataframe(self):
        """Test that preprocess_prediction_data returns DataFrame."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        preprocessor.fit(df)

        result = preprocess_prediction_data(df, preprocessor)

        assert isinstance(result, pd.DataFrame)

    def test_preprocess_prediction_data_no_target_column(self):
        """Test that prediction data doesn't include target."""
        df = fetch_call_samples(limit=10)
        preprocessor = OptionPreprocessor()
        preprocessor.fit(df)

        result = preprocess_prediction_data(df, preprocessor)

        assert "delta" not in result.columns

    def test_preprocess_prediction_data_same_features_as_training(self):
        """Test that prediction data has same features as training."""
        df = fetch_call_samples(limit=20)

        # Fit preprocessor on some data
        preprocessor = OptionPreprocessor()
        train_df = preprocessor.fit_transform(df.iloc[:10], include_target=False)

        # Transform prediction data
        pred_df = preprocess_prediction_data(df.iloc[10:], preprocessor)

        assert list(train_df.columns) == list(pred_df.columns)
