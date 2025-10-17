"""Data loading utilities for fetching training data from PostgreSQL."""
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.db import get_db_cursor


def load_call_samples_query() -> str:
    """
    Load the call samples SQL query from file.

    Returns:
        str: SQL query string
    """
    query_path = Path(__file__).parent / 'call_samples.sql'
    with open(query_path, 'r') as f:
        return f.read()


def fetch_call_samples(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch call option samples from the database.

    This loads SPXW call options with associated volatility indicators (VIX, VIX9D, VVIX, SKEW).
    The query filters for:
    - Sample data (sample = true)
    - SPXW root symbol
    - Moneyness >= 0.99 (near or at-the-money)
    - Call options only
    - DTE <= 9 days

    Args:
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
            - contract_type: 'CALL'
            - valid_time: timestamp
            - vix, vix9d, vvix, skew: volatility indicators
    """
    query = load_call_samples_query()

    if limit:
        query += f"\nLIMIT {limit}"

    with get_db_cursor() as cur:
        cur.execute(query)
        results = cur.fetchall()

    df = pd.DataFrame(results)

    # Convert timestamp columns to datetime
    if 'valid_time' in df.columns:
        df['valid_time'] = pd.to_datetime(df['valid_time'])

    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = 'delta',
    drop_cols: Optional[list] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from the dataframe.

    Args:
        df: Input dataframe
        target_col: Name of the target column (default: 'delta')
        drop_cols: Additional columns to drop from features (default: drops common non-feature columns)

    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    if drop_cols is None:
        # Default columns to exclude from features
        drop_cols = [
            'delta',  # target variable
            'valid_time',  # timestamp
            'contract_type',  # constant for this query
        ]

    # Remove rows with missing target
    df = df.dropna(subset=[target_col])

    # Separate features and target
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    return X, y


if __name__ == '__main__':
    # Example usage
    print("Fetching call samples...")
    df = fetch_call_samples(limit=10)
    print(f"\nLoaded {len(df)} samples")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")

    X, y = prepare_features_target(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {X.columns.tolist()}")
