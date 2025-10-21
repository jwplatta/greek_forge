"""Data preprocessing for model training and inference."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class OptionPreprocessor:
    """
    Preprocessor for option data.

    Handles label encoding and feature preparation for both training and inference.
    """

    def __init__(self):
        self.contract_type_encoder = LabelEncoder()
        self.feature_columns = [
            "dte",
            "moneyness",
            "mark",
            "strike",
            "underlying_price",
            "vix9d",
            "vvix",
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "OptionPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            df: Training dataframe with contract_type column

        Returns:
            self: Fitted preprocessor
        """
        if "contract_type" in df.columns:
            self.contract_type_encoder.fit(df["contract_type"])

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Transform data for model input.

        Args:
            df: Input dataframe
            include_target: Whether to include the delta column (False for inference)

        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()

        if "contract_type" in df.columns:
            df["contract_type"] = self.contract_type_encoder.transform(
                df["contract_type"]
            )

        columns_to_keep = self.feature_columns.copy()
        if include_target and "delta" in df.columns:
            columns_to_keep.append("delta")

        columns_to_keep = [col for col in columns_to_keep if col in df.columns]

        return df[columns_to_keep]

    def fit_transform(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Input dataframe
            include_target: Whether to include the delta column

        Returns:
            Transformed dataframe
        """
        return self.fit(df).transform(df, include_target=include_target)


def balance_delta_samples(
    df: pd.DataFrame,
    n_bins: int = 10,
    samples_per_bin: int = 2000,
    target_col: str = "delta",
    random_state: int = 42,
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
        random_state: Random seed for reproducibility

    Returns:
        pd.DataFrame: Balanced dataframe with resampled data
    """
    df = df.copy()
    df["delta_bin"] = pd.cut(df[target_col], bins=n_bins, labels=False)
    balanced_df = (
        df.groupby("delta_bin", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=samples_per_bin, replace=True, random_state=random_state
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # Drop delta_bin column if it exists (may not exist with include_groups=False)
    if "delta_bin" in balanced_df.columns:
        balanced_df = balanced_df.drop("delta_bin", axis=1)

    return balanced_df


def preprocess_training_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    balance_samples: bool = True,
    n_bins: int = 10,
    samples_per_bin: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, OptionPreprocessor]:
    """
    Complete preprocessing pipeline for training data.

    Args:
        df: Raw dataframe from loader
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        balance_samples: Whether to balance delta samples via binning
        n_bins: Number of bins for balancing (default: 10)
        samples_per_bin: Samples per bin when balancing (default: 2000)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    df = df.dropna(subset=["delta"])

    # Balance samples if requested
    if balance_samples:
        df = balance_delta_samples(
            df,
            n_bins=n_bins,
            samples_per_bin=samples_per_bin,
            random_state=random_state,
        )

    preprocessor = OptionPreprocessor()
    preprocessor.fit(df)

    df_transformed = preprocessor.transform(df, include_target=True)

    X = df_transformed.drop(columns=["delta"])
    y = df_transformed["delta"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor


def preprocess_prediction_data(
    df: pd.DataFrame, preprocessor: OptionPreprocessor
) -> pd.DataFrame:
    """
    Preprocess new data for predictions.

    Args:
        df: Raw dataframe with same structure as training data
        preprocessor: Fitted preprocessor from training

    Returns:
        Transformed feature dataframe ready for prediction
    """
    return preprocessor.transform(df, include_target=False)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import fetch_call_samples

    print("Fetching training data...")
    df = fetch_call_samples()
    print(f"Loaded {len(df)} samples")

    print("\nPreprocessing training data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(df)

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nFeature columns: {X_train.columns.tolist()}")
    print("\nTarget distribution (train):")
    print(y_train.describe())

    print("\nContract type encoding:")
    if hasattr(preprocessor.contract_type_encoder, "classes_"):
        print(f"Classes: {preprocessor.contract_type_encoder.classes_}")
