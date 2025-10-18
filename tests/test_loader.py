"""Tests for data loader module."""

import pandas as pd

from src.data.loader import (
    fetch_call_samples,
    fetch_put_samples,
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
