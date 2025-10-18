"""Tests for data loader module."""

import pandas as pd

from src.data.loader import (
    fetch_call_samples,
    fetch_put_samples,
    balance_delta_samples,
    prepare_features_target,
)


class TestFetchOptionSamples:
    """Test option data fetching."""

    def test_fetch_call_samples_returns_dataframe(self):
        """Test that fetch_call_samples returns a DataFrame."""
        df = fetch_call_samples(limit=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_put_samples_returns_dataframe(self):
        """Test that fetch_put_samples returns a DataFrame."""
        df = fetch_put_samples(limit=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_option_samples_has_required_columns(self):
        """Test that fetched data has required columns."""
        df = fetch_call_samples(limit=5)

        required_columns = [
            "dte",
            "moneyness",
            "mark",
            "underlying_price",
            "delta",
            "strike",
            "contract_type",
            "valid_time",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_fetch_call_samples_contract_type(self):
        """Test that fetch_call_samples only returns CALL options."""
        df = fetch_call_samples(limit=5)
        assert (df["contract_type"] == "CALL").all()

    def test_fetch_put_samples_contract_type(self):
        """Test that fetch_put_samples only returns PUT options."""
        df = fetch_put_samples(limit=5)
        assert (df["contract_type"] == "PUT").all()

    def test_numeric_columns_are_floats(self):
        """Test that numeric columns are float type."""
        df = fetch_call_samples(limit=5)

        numeric_columns = ["dte", "moneyness", "mark", "delta", "strike"]

        for col in numeric_columns:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), (
                    f"Column {col} should be numeric"
                )

    def test_limit_parameter_works(self):
        """Test that limit parameter constrains result size."""
        limit = 5
        df = fetch_call_samples(limit=limit)
        assert len(df) <= limit


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


class TestPrepareFeatureTarget:
    """Test feature/target preparation."""

    def test_prepare_features_target_returns_correct_types(self):
        """Test that prepare_features_target returns DataFrame and Series."""
        df = fetch_call_samples(limit=50)
        X, y = prepare_features_target(df, balance_samples=False)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_prepare_features_target_has_correct_features(self):
        """Test that features match expected columns."""
        df = fetch_call_samples(limit=50)
        X, y = prepare_features_target(df, balance_samples=False)

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

        assert list(X.columns) == expected_features

    def test_prepare_features_target_removes_delta_from_features(self):
        """Test that delta is not in features."""
        df = fetch_call_samples(limit=50)
        X, y = prepare_features_target(df, balance_samples=False)

        assert "delta" not in X.columns

    def test_prepare_features_target_target_is_delta(self):
        """Test that target is delta values."""
        df = fetch_call_samples(limit=50)
        X, y = prepare_features_target(df, balance_samples=False)

        assert y.name == "delta"

    def test_prepare_features_target_no_missing_values(self):
        """Test that there are no missing values in features or target."""
        df = fetch_call_samples(limit=50)
        X, y = prepare_features_target(df, balance_samples=False)

        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_prepare_features_target_with_balancing(self):
        """Test that balancing option works."""
        df = fetch_call_samples()
        X_balanced, y_balanced = prepare_features_target(
            df, balance_samples=True, n_bins=5, samples_per_bin=100
        )

        # Should have 5 bins * 100 samples = 500 rows
        assert len(X_balanced) == 500
        assert len(y_balanced) == 500
