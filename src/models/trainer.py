"""Model training utilities."""

from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np


class DeltaPredictor:
    """
    Wrapper for HistGradientBoostingRegressor with defaults tuned for delta prediction.
    """

    def __init__(self, **params):
        """
        Initialize model with hyperparameters.

        Args:
            **params: Hyperparameters for HistGradientBoostingRegressor
        """
        default_params = {
            "max_iter": 125,
            "min_samples_leaf": 10,
            "learning_rate": 0.15,
            "max_depth": 10,
            "categorical_features": None,
        }

        default_params.update(params)

        self.model = HistGradientBoostingRegressor(**default_params)
        self.params = default_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DeltaPredictor":
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target values (delta)

        Returns:
            self: Fitted predictor
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted delta values
        """
        return self.model.predict(X)

    def get_params(self) -> Dict:
        """Get model hyperparameters."""
        return self.params


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict] = None
) -> DeltaPredictor:
    """
    Train a delta prediction model.

    Args:
        X_train: Training features
        y_train: Training target
        params: Optional hyperparameters

    Returns:
        Trained DeltaPredictor
    """
    if params is None:
        params = {}

    predictor = DeltaPredictor(**params)
    predictor.fit(X_train, y_train)

    return predictor


def train_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "neg_mean_absolute_error",
) -> Tuple[DeltaPredictor, Dict]:
    """
    Train model with cross-validation.

    Args:
        X_train: Training features
        y_train: Training target
        params: Optional hyperparameters
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Tuple of (trained model, cv scores dict)
    """
    if params is None:
        params = {}

    predictor = DeltaPredictor(**params)

    print(f"Running {cv}-fold cross-validation...")
    cv_scores = cross_val_score(
        predictor.model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )

    if scoring.startswith("neg_"):
        cv_scores = -cv_scores

    cv_results = {
        "scores": cv_scores.tolist(),
        "mean": cv_scores.mean(),
        "std": cv_scores.std(),
        "metric": scoring.replace("neg_", ""),
    }

    print(f"{scoring.replace('neg_', '').upper()} scores: {cv_scores}")
    print(f"Mean: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

    print("\nTraining on full training set...")
    predictor.fit(X_train, y_train)

    return predictor, cv_results


if __name__ == "__main__":
    from src.data.loader import fetch_call_samples
    from src.data.preprocessor import preprocess_training_data

    print("Loading and preprocessing data...")
    df = fetch_call_samples()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(df)

    print("\nTraining model with cross-validation...")
    model, cv_results = train_with_cv(X_train, y_train)

    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)

    print("\nFirst 10 predictions:")
    for i in range(min(10, len(y_pred))):
        print(f"  Actual: {y_test.iloc[i]:.4f}, Predicted: {y_pred[i]:.4f}")
