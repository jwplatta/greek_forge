"""Model persistence and versioning utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib

from src.utils.constants import CONTRACT_TYPE_CALL, ContractType
from src.utils.logger import get_logger

logger = get_logger()


def get_model_dir(
    contract_type: ContractType, version: str, base_dir: Optional[Path] = None
) -> Path:
    """
    Get the directory path for a specific model version.

    Args:
        contract_type: 'CALL' or 'PUT'
        version: Model version (e.g., '1.0.0')
        base_dir: Optional base directory (defaults to project_root/models)

    Returns:
        Path to model directory
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "models"

    contract_dir = "calls" if contract_type == CONTRACT_TYPE_CALL else "puts"
    model_dir = base_dir / contract_dir / f"v{version}"

    return model_dir


def save_model(
    model,
    preprocessor,
    contract_type: ContractType,
    version: str,
    metrics: Dict,
    hyperparameters: Dict,
    feature_names: Optional[list] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Save trained model with all artifacts.

    Args:
        model: Trained model object
        preprocessor: Fitted preprocessor
        contract_type: 'CALL' or 'PUT'
        version: Semantic version string (e.g., '1.0.0')
        metrics: Dictionary of evaluation metrics
        hyperparameters: Model hyperparameters
        feature_names: List of feature column names
        base_dir: Optional base directory

    Returns:
        Path to saved model directory
    """
    model_dir = get_model_dir(contract_type, version, base_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    preprocessor_path = model_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to: {preprocessor_path}")

    metadata = {
        "contract_type": contract_type,
        "version": version,
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "model_type": type(model).__name__,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

    if feature_names is not None:
        feature_path = model_dir / "feature_names.json"
        with open(feature_path, "w") as f:
            json.dump({"features": feature_names}, f, indent=2)
        logger.info(f"Feature names saved to: {feature_path}")

    logger.info(f"Model artifacts saved to: {model_dir}")
    return model_dir


def load_model(
    contract_type: ContractType, version: str, base_dir: Optional[Path] = None
) -> Tuple:
    """
    Load trained model with all artifacts.

    Args:
        contract_type: 'CALL' or 'PUT'
        version: Model version to load
        base_dir: Optional base directory

    Returns:
        Tuple of (model, preprocessor, metadata)
    """
    model_dir = get_model_dir(contract_type, version, base_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    preprocessor_path = model_dir / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    metadata_path = model_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    logger.info(f"Loaded model version {version} for {contract_type} options")
    logger.info(f"  Created: {metadata.get('created_at', 'unknown')}")
    logger.info(f"  Model type: {metadata.get('model_type', 'unknown')}")

    return model, preprocessor, metadata


def list_model_versions(
    contract_type: ContractType, base_dir: Optional[Path] = None
) -> list:
    """
    List all available model versions for a contract type.

    Args:
        contract_type: 'CALL' or 'PUT'
        base_dir: Optional base directory

    Returns:
        List of version strings
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "models"

    contract_dir = "calls" if contract_type == CONTRACT_TYPE_CALL else "puts"
    contract_path = base_dir / contract_dir

    if not contract_path.exists():
        return []

    versions = []
    for version_dir in contract_path.iterdir():
        if version_dir.is_dir() and version_dir.name.startswith("v"):
            version = version_dir.name[1:]
            versions.append(version)

    return sorted(versions)


def get_latest_version(
    contract_type: ContractType, base_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Get the latest model version for a contract type.

    Args:
        contract_type: 'CALL' or 'PUT'
        base_dir: Optional base directory

    Returns:
        Latest version string or None if no models exist
    """
    versions = list_model_versions(contract_type, base_dir)
    if not versions:
        return None

    return versions[-1]


if __name__ == "__main__":
    import argparse
    from src.data.loader import fetch_option_samples
    from src.data.preprocessor import preprocess_training_data
    from src.models.trainer import train_with_cv
    from src.models.evaluator import evaluate_model, generate_evaluation_report

    parser = argparse.ArgumentParser(
        description="Train and save option delta prediction model"
    )
    from src.utils.constants import VALID_CONTRACT_TYPES, CONTRACT_TYPE_CALL

    parser.add_argument(
        "--contract-type",
        type=str,
        choices=VALID_CONTRACT_TYPES,
        default=CONTRACT_TYPE_CALL,
        help="Type of option contract (CALL or PUT)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version to save (default: 1.0.0)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    args = parser.parse_args()

    logger.info(f"Training {args.contract_type} option delta prediction model...")
    logger.info(f"Version: {args.version}")
    logger.info("=" * 60)

    logger.info("Loading and preprocessing data...")
    df = fetch_option_samples(contract_type=args.contract_type)
    logger.info(f"Loaded {len(df)} samples")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(df)
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    logger.info("=" * 60)
    logger.info(f"Training model with {args.cv_folds}-fold cross-validation...")
    model, cv_results = train_with_cv(X_train, y_train, cv=args.cv_folds)

    logger.info(f"Cross-validation {cv_results['metric']}:")
    logger.info(f"  Mean: {cv_results['mean']:.6f}")
    logger.info(f"  Std: {cv_results['std']:.6f}")
    logger.info(f"  Scores: {cv_results['scores']}")

    logger.info("=" * 60)
    logger.info("Evaluating model on test set...")
    evaluation = evaluate_model(model, X_train, X_test, y_train, y_test)
    generate_evaluation_report(evaluation)

    logger.info("=" * 60)
    logger.info("Saving model...")
    metrics = {
        "cv_mean_mae": cv_results["mean"],
        "cv_std_mae": cv_results["std"],
        "test_mae": evaluation["test_metrics"]["mae"],
        "test_rmse": evaluation["test_metrics"]["rmse"],
        "test_r2": evaluation["test_metrics"]["r2"],
        "train_mae": evaluation["train_metrics"]["mae"],
    }

    model_dir = save_model(
        model=model,
        preprocessor=preprocessor,
        contract_type=args.contract_type,
        version=args.version,
        metrics=metrics,
        hyperparameters=model.get_params(),
        feature_names=X_train.columns.tolist(),
    )

    logger.info("=" * 60)
    logger.info("Model training complete!")
    logger.info(f"Model saved to: {model_dir}")
    logger.info("To use this model:")
    logger.info("  from src.api.predictor import Predictor")
    logger.info(
        f"  predictor = Predictor.load('{args.contract_type}', '{args.version}')"
    )
