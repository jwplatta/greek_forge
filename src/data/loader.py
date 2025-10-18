"""Data loading utilities for fetching training data from PostgreSQL."""

from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from src.utils.db import get_db_cursor


def load_option_samples_query() -> str:
    """
    Load the option samples SQL query from file.

    Returns:
        str: SQL query string (parameterized with contract_type)
    """
    query_path = Path(__file__).parent / "option_samples.sql"
    with open(query_path, "r") as f:
        return f.read()


def fetch_option_samples(
    contract_type: Literal["CALL", "PUT"] = "CALL", limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch option samples from the database.

    This loads SPXW options with associated volatility indicators (VIX, VIX9D, VVIX, SKEW).
    The query filters for:
    - Sample data (sample = true)
    - SPXW root symbol
    - Moneyness >= 0.99 (near or at-the-money)
    - Specified contract type (CALL or PUT)
    - DTE <= 9 days

    Args:
        contract_type: 'CALL' or 'PUT' (default: 'CALL')
        limit: Optional limit on number of rows to fetch

    Returns:
        pd.DataFrame: DataFrame with columns:
            - dte: days to expiration
            - moneyness: strike / underlying_price
            - spread: ask - bid
            - ask, bid, mark: option prices
            - underlying_price: underlying asset price
            - delta: option delta (target variable)
            - strike: option strike price
            - contract_type: 'CALL' or 'PUT'
            - valid_time: timestamp
            - vix, vix9d, vvix, skew: volatility indicators
    """
    query = load_option_samples_query()

    if limit:
        query += f"\nLIMIT {limit}"

    with get_db_cursor() as cur:
        cur.execute(query, {"contract_type": contract_type})
        results = cur.fetchall()

    df = pd.DataFrame(results)

    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])

    numeric_cols = [
        "dte",
        "moneyness",
        "mark",
        "underlying_price",
        "delta",
        "strike",
        "vix",
        "vix9d",
        "vvix",
        "skew",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_call_samples(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch call option samples from the database.

    Convenience wrapper around fetch_option_samples for call options.

    Args:
        limit: Optional limit on number of rows to fetch

    Returns:
        pd.DataFrame: DataFrame with call option samples
    """
    return fetch_option_samples(contract_type="CALL", limit=limit)


def fetch_put_samples(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch put option samples from the database.

    Convenience wrapper around fetch_option_samples for put options.

    Args:
        limit: Optional limit on number of rows to fetch

    Returns:
        pd.DataFrame: DataFrame with put option samples
    """
    return fetch_option_samples(contract_type="PUT", limit=limit)


def balance_delta_samples(
    df: pd.DataFrame,
    n_bins: int = 10,
    samples_per_bin: int = 2000,
    target_col: str = "delta",
) -> pd.DataFrame:
    """
    Balance dataset by binning delta values and resampling each bin.

    This addresses the class imbalance problem where most delta values are near 0.
    Creates equal-sized bins and resamples each to have the same number of samples.

    Args:
        df: Input dataframe
        n_bins: Number of bins to create (default: 10)
        samples_per_bin: Number of samples per bin (default: 2000)
        target_col: Name of the target column to bin (default: 'delta')

    Returns:
        pd.DataFrame: Balanced dataframe with resampled data
    """
    df["delta_bin"] = pd.cut(df[target_col], bins=n_bins, labels=False)
    balanced_df = (
        df.groupby("delta_bin", group_keys=False)
        .apply(lambda x: x.sample(n=samples_per_bin, replace=True))
        .reset_index(drop=True)
    )

    balanced_df = balanced_df.drop("delta_bin", axis=1)

    return balanced_df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = "delta",
    feature_cols: Optional[list] = None,
    balance_samples: bool = True,
    n_bins: int = 10,
    samples_per_bin: int = 2000,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from the dataframe.

    Args:
        df: Input dataframe
        target_col: Name of the target column (default: 'delta')
        feature_cols: List of feature columns to use. If None, uses default features
        balance_samples: Whether to balance samples via delta binning (default: True)
        n_bins: Number of bins for balancing (default: 10)
        samples_per_bin: Samples per bin when balancing (default: 2000)

    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    df = df.dropna(subset=[target_col])

    if balance_samples:
        df = balance_delta_samples(
            df, n_bins=n_bins, samples_per_bin=samples_per_bin, target_col=target_col
        )

    if feature_cols is None:
        feature_cols = [
            "dte",
            "moneyness",
            "mark",
            "strike",
            "underlying_price",
            "vix9d",
            "vvix",
            "skew",
        ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


if __name__ == "__main__":
    # Example usage
    print("Fetching call samples...")
    df_calls = fetch_call_samples()
    print(f"\nLoaded {len(df_calls)} call samples")
    print(f"\nColumns: {df_calls.columns.tolist()}")

    print("\n" + "=" * 60)
    print("Delta distribution before balancing:")
    print(df_calls["delta"].describe())
    print("\nDelta value counts (binned):")
    df_calls["temp_bin"] = pd.cut(df_calls["delta"], bins=10, labels=False)
    print(df_calls["temp_bin"].value_counts().sort_index())
    df_calls = df_calls.drop("temp_bin", axis=1)

    print("\n" + "=" * 60)
    print("Preparing features and target (with balancing)...")
    X, y = prepare_features_target(df_calls, balance_samples=True)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {X.columns.tolist()}")

    print("\n" + "=" * 60)
    print("Delta distribution after balancing:")
    print(y.describe())

    print("\n" + "=" * 60)
    print("Testing without balancing...")
    X_unbalanced, y_unbalanced = prepare_features_target(
        df_calls, balance_samples=False
    )
    print(f"Unbalanced shape: {X_unbalanced.shape}")
    print(f"Balanced shape: {X.shape}")
