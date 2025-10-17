"""Data loading utilities for fetching training data from PostgreSQL."""
import os
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
    query_path = Path(__file__).parent / 'option_samples.sql'
    with open(query_path, 'r') as f:
        return f.read()


def fetch_option_samples(
    contract_type: Literal['CALL', 'PUT'] = 'CALL',
    limit: Optional[int] = None
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
        cur.execute(query, {'contract_type': contract_type})
        results = cur.fetchall()

    df = pd.DataFrame(results)

    # Convert timestamp columns to datetime
    if 'valid_time' in df.columns:
        df['valid_time'] = pd.to_datetime(df['valid_time'])

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
    return fetch_option_samples(contract_type='CALL', limit=limit)


def fetch_put_samples(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch put option samples from the database.

    Convenience wrapper around fetch_option_samples for put options.

    Args:
        limit: Optional limit on number of rows to fetch

    Returns:
        pd.DataFrame: DataFrame with put option samples
    """
    return fetch_option_samples(contract_type='PUT', limit=limit)


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
    df_calls = fetch_call_samples(limit=5)
    print(f"\nLoaded {len(df_calls)} call samples")
    print(f"\nColumns: {df_calls.columns.tolist()}")
    print(f"\nFirst few rows:\n{df_calls.head()}")

    print("\n" + "="*60)
    print("Fetching put samples...")
    df_puts = fetch_put_samples(limit=5)
    print(f"\nLoaded {len(df_puts)} put samples")
    print(f"\nFirst few rows:\n{df_puts.head()}")

    print("\n" + "="*60)
    print("Preparing features and target...")
    X, y = prepare_features_target(df_calls)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {X.columns.tolist()}")
