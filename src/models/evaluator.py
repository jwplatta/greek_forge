"""Model evaluation and metrics."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        X_train: Optional training features (for training metrics)
        y_train: Optional training target (for training metrics)

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred_test = model.predict(X_test)

    test_metrics = {
        "mae": mean_absolute_error(y_test, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "r2": r2_score(y_test, y_pred_test),
    }

    metrics = {"test": test_metrics}

    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)

        train_metrics = {
            "mae": mean_absolute_error(y_train, y_pred_train),
            "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "r2": r2_score(y_train, y_pred_train),
        }

        metrics["train"] = train_metrics

    return metrics


def calculate_feature_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation feature importance.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        n_repeats: Number of times to permute each feature
        random_state: Random seed

    Returns:
        DataFrame with feature importance scores
    """
    print("Calculating feature importance (this may take a minute)...")

    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return feature_importance


def analyze_predictions(
    y_true: pd.Series, y_pred: np.ndarray, X: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Analyze prediction errors and patterns.

    Args:
        y_true: True target values
        y_pred: Predicted values
        X: Optional feature matrix for error analysis

    Returns:
        Dictionary with error analysis
    """
    errors = np.abs(y_true - y_pred)

    analysis = {
        "mean_error": errors.mean(),
        "median_error": np.median(errors),
        "max_error": errors.max(),
        "min_error": errors.min(),
        "std_error": errors.std(),
        "percentile_90": np.percentile(errors, 90),
        "percentile_95": np.percentile(errors, 95),
        "percentile_99": np.percentile(errors, 99),
    }

    error_bins = {
        "errors_under_0.01": (errors < 0.01).sum(),
        "errors_0.01_to_0.05": ((errors >= 0.01) & (errors < 0.05)).sum(),
        "errors_0.05_to_0.1": ((errors >= 0.05) & (errors < 0.1)).sum(),
        "errors_over_0.1": (errors >= 0.1).sum(),
    }

    analysis["error_bins"] = error_bins

    return analysis


def print_evaluation_report(
    metrics: Dict,
    feature_importance: Optional[pd.DataFrame] = None,
    error_analysis: Optional[Dict] = None,
):
    """
    Print a formatted evaluation report.

    Args:
        metrics: Metrics from evaluate_model()
        feature_importance: Optional feature importance DataFrame
        error_analysis: Optional error analysis from analyze_predictions()
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    if "test" in metrics:
        print("\nTest Set Performance:")
        print(f"  MAE:  {metrics['test']['mae']:.6f}")
        print(f"  RMSE: {metrics['test']['rmse']:.6f}")
        print(f"  R²:   {metrics['test']['r2']:.6f}")

    if "train" in metrics:
        print("\nTraining Set Performance:")
        print(f"  MAE:  {metrics['train']['mae']:.6f}")
        print(f"  RMSE: {metrics['train']['rmse']:.6f}")
        print(f"  R²:   {metrics['train']['r2']:.6f}")

        if "test" in metrics:
            mae_diff = abs(metrics["train"]["mae"] - metrics["test"]["mae"])
            if mae_diff > 0.01:
                print(f"\n⚠️  Possible overfitting detected (MAE diff: {mae_diff:.6f})")

    if feature_importance is not None:
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)
        print(feature_importance.to_string(index=False))

    if error_analysis is not None:
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)
        print("\nError Statistics:")
        print(f"  Mean:   {error_analysis['mean_error']:.6f}")
        print(f"  Median: {error_analysis['median_error']:.6f}")
        print(f"  Std:    {error_analysis['std_error']:.6f}")
        print(f"  Max:    {error_analysis['max_error']:.6f}")

        print("\nError Percentiles:")
        print(f"  90th: {error_analysis['percentile_90']:.6f}")
        print(f"  95th: {error_analysis['percentile_95']:.6f}")
        print(f"  99th: {error_analysis['percentile_99']:.6f}")

        print("\nError Distribution:")
        bins = error_analysis["error_bins"]
        total = sum(bins.values())
        for bin_name, count in bins.items():
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {bin_name}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # NOTE: Example usage
    from src.data.loader import fetch_call_samples
    from src.data.preprocessor import preprocess_training_data
    from src.models.trainer import train_with_cv

    print("Loading and preprocessing data...")
    df = fetch_call_samples()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(df)

    print("\nTraining model...")
    model, cv_results = train_with_cv(X_train, y_train, cv=3)

    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)

    print("\nCalculating feature importance...")
    feature_importance = calculate_feature_importance(
        model.model,
        X_train,
        y_train,
        n_repeats=5,
    )

    print("\nAnalyzing prediction errors...")
    y_pred = model.predict(X_test)
    error_analysis = analyze_predictions(y_test, y_pred, X_test)

    print_evaluation_report(metrics, feature_importance, error_analysis)
