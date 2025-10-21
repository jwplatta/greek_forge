"""Hyperparameter optimization using Optuna."""

from typing import Dict, Optional

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src.utils.logger import get_logger

logger = get_logger()


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
    study_name: Optional[str] = None,
) -> Dict:
    """
    Optimize hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random seed
        study_name: Optional name for the Optuna study

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        """Optuna objective function."""
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
            "random_state": random_state,
        }

        model = HistGradientBoostingRegressor(**params)

        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )

        return -scores.mean()

    if study_name is None:
        study_name = "delta_prediction_optimization"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best trial:")
    logger.info(f"  MAE: {study.best_value:.6f}")
    logger.info(f"  Params: {study.best_params}")

    return study.best_params


def optimize_with_pruning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    cv: int = 5,
    random_state: int = 42,
    timeout: Optional[int] = None,
) -> Dict:
    """
    Optimize with pruning for faster trials.

    Pruning stops unpromising trials early to save computation time.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random seed
        timeout: Maximum time in seconds (None for no limit)

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        """Optuna objective with pruning."""
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "random_state": random_state,
        }

        model = HistGradientBoostingRegressor(**params)

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            score = -model.score(X_fold_val, y_fold_val)

            scores.append(score)

            trial.report(score, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(scores) / len(scores)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    logger.info(f"Starting optimization with pruning ({n_trials} trials max)...")
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )

    logger.info("Best trial:")
    logger.info(f"  Score: {study.best_value:.6f}")
    logger.info(f"  Params: {study.best_params}")
    logger.info(f"  Completed trials: {len(study.trials)}")
    logger.info(
        f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )

    return study.best_params


if __name__ == "__main__":
    # NOTE: Example usage
    from src.data.loader import fetch_call_samples
    from src.data.preprocessor import preprocess_training_data

    logger.info("Loading and preprocessing data...")
    df = fetch_call_samples()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(df)

    logger.info("Running hyperparameter optimization (20 trials for demo)...")
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=20, cv=5)

    logger.info("=" * 60)
    logger.info("Best hyperparameters found:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
