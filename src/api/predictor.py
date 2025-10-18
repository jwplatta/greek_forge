"""Prediction service for loading models and generating predictions."""

from typing import Dict, List, Literal, Optional
import pandas as pd
import numpy as np

from src.utils.model_io import load_model, get_latest_version


class Predictor:
    """
    Prediction service that loads a trained model and generates predictions.

    This class encapsulates model loading and preprocessing for inference.
    """

    def __init__(self, model, preprocessor, metadata: Dict):
        """
        Initialize predictor with loaded artifacts.

        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            metadata: Model metadata
        """
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.contract_type = metadata.get("contract_type")
        self.version = metadata.get("version")

    @classmethod
    def load(
        cls, contract_type: Literal["CALL", "PUT"], version: Optional[str] = None
    ) -> "Predictor":
        """
        Load a predictor from saved model artifacts.

        Args:
            contract_type: 'CALL' or 'PUT'
            version: Model version to load. If None, loads latest version.

        Returns:
            Initialized Predictor instance
        """
        if version is None:
            version = get_latest_version(contract_type)
            if version is None:
                raise FileNotFoundError(f"No models found for {contract_type} options")
            print(f"Loading latest version: {version}")

        model, preprocessor, metadata = load_model(contract_type, version)
        return cls(model, preprocessor, metadata)

    def predict_single(self, input_data: Dict) -> float:
        """
        Generate prediction for a single option.

        Args:
            input_data: Dictionary with feature values
                {
                    'dte': int,
                    'moneyness': float,
                    'mark': float,
                    'strike': float,
                    'underlying_price': float,
                    'vix9d': float,
                    'vvix': float,
                    'skew': float
                }

        Returns:
            Predicted delta value
        """
        df = pd.DataFrame([input_data])
        X = self.preprocessor.transform(df, include_target=False)
        prediction = self.model.predict(X)

        return float(prediction[0])

    def predict_batch(self, input_data: List[Dict]) -> np.ndarray:
        """
        Generate predictions for multiple options.

        Args:
            input_data: List of dictionaries with feature values

        Returns:
            Array of predicted delta values
        """
        df = pd.DataFrame(input_data)
        X = self.preprocessor.transform(df, include_target=False)
        predictions = self.model.predict(X)

        return predictions

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from a DataFrame.

        Args:
            df: DataFrame with feature columns

        Returns:
            Array of predicted delta values
        """
        X = self.preprocessor.transform(df, include_target=False)
        predictions = self.model.predict(X)

        return predictions

    def get_feature_names(self) -> List[str]:
        """Get the list of expected feature names."""
        return self.preprocessor.feature_columns

    def get_info(self) -> Dict:
        """Get model information."""
        return {
            "contract_type": self.contract_type,
            "version": self.version,
            "created_at": self.metadata.get("created_at"),
            "model_type": self.metadata.get("model_type"),
            "metrics": self.metadata.get("metrics", {}),
            "hyperparameters": self.metadata.get("hyperparameters", {}),
            "feature_names": self.get_feature_names(),
        }


if __name__ == "__main__":
    print("Loading predictor for CALL options...")

    try:
        predictor = Predictor.load(contract_type="CALL")

        print("\nModel info:")
        info = predictor.get_info()
        print(f"  Version: {info['version']}")
        print(f"  Created: {info['created_at']}")
        print(f"  Model type: {info['model_type']}")
        print(f"  Metrics: {info['metrics']}")

        print("\nExpected features:")
        for feature in predictor.get_feature_names():
            print(f"  - {feature}")

        print("\n" + "=" * 60)
        print("Testing single prediction...")
        input_data = {
            "dte": 5,
            "moneyness": 0.99,
            "mark": 10.5,
            "strike": 6000.0,
            "underlying_price": 6060.0,
            "vix9d": 15.0,
            "vvix": 90.0,
            "skew": 140.0,
        }

        delta = predictor.predict_single(input_data)
        print(f"\nInput: {input_data}")
        print(f"Predicted delta: {delta:.4f}")

        print("\n" + "=" * 60)
        print("Testing batch prediction...")
        batch_data = [
            {**input_data, "strike": 5900.0},
            {**input_data, "strike": 6020.0},
            {**input_data, "strike": 6150.0},
        ]

        deltas = predictor.predict_batch(batch_data)
        print("\nBatch predictions:")
        for i, (data, delta) in enumerate(zip(batch_data, deltas)):
            print(f"  {i + 1}. Strike: {data['strike']:.0f} -> Delta: {delta:.4f}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nNo models found. Train and save a model first using:")
        print("  uv run python src/utils/model_io.py")
