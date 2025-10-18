"""Tests for FastAPI application."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor

from src.api.app import app
from src.utils.model_io import save_model


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing."""
    predictor = MagicMock()
    predictor.version = "1.0.0"
    predictor.predict_single.return_value = 0.65
    predictor.predict_batch.return_value = np.array([0.65, 0.72, 0.58])
    return predictor


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory with a saved model."""
    # Create a simple model
    model = HistGradientBoostingRegressor(random_state=42)
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8]]).T
    y = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    model.fit(X.reshape(-1, 1), y)

    # Save the model
    preprocessor = MagicMock()
    metrics = {"test_mae": 0.05, "test_rmse": 0.08, "test_r2": 0.95}
    hyperparameters = {"max_iter": 100}

    save_model(
        model=model,
        preprocessor=preprocessor,
        contract_type="CALL",
        version="1.0.0",
        metrics=metrics,
        hyperparameters=hyperparameters,
        base_dir=tmp_path,
    )

    return tmp_path


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Greek Forge API"


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_status(self, client):
        """Test that health check returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] == "healthy"


class TestPredictDeltaEndpoint:
    """Test single delta prediction endpoint."""

    @patch("src.api.app.predictors")
    def test_single_prediction_success(self, mock_predictors, client, mock_predictor):
        """Test successful single prediction."""
        mock_predictors.__getitem__.return_value = mock_predictor
        mock_predictors.__contains__.return_value = True

        request_data = {
            "contract_type": "CALL",
            "features": {
                "dte": 5,
                "moneyness": 0.99,
                "mark": 10.5,
                "strike": 6000.0,
                "underlying_price": 6060.0,
                "vix9d": 15.0,
                "vvix": 90.0,
                "skew": 140.0,
            },
            "version": "latest",
        }

        response = client.post("/predict_delta", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "delta" in data
        assert "contract_type" in data
        assert "model_version" in data
        assert data["delta"] == 0.65
        assert data["contract_type"] == "CALL"

    def test_invalid_contract_type(self, client):
        """Test prediction with invalid contract type."""
        request_data = {
            "contract_type": "INVALID",
            "features": {
                "dte": 5,
                "moneyness": 0.99,
                "mark": 10.5,
                "strike": 6000.0,
                "underlying_price": 6060.0,
                "vix9d": 15.0,
                "vvix": 90.0,
                "skew": 140.0,
            },
        }

        response = client.post("/predict_delta", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_missing_required_field(self, client):
        """Test prediction with missing required field."""
        request_data = {
            "contract_type": "CALL",
            "features": {
                "dte": 5,
                "moneyness": 0.99,
                # Missing other required fields
            },
        }

        response = client.post("/predict_delta", json=request_data)
        assert response.status_code == 422  # Validation error


class TestPredictDeltasEndpoint:
    """Test batch delta prediction endpoint."""

    @patch("src.api.app.predictors")
    def test_batch_prediction_success(self, mock_predictors, client, mock_predictor):
        """Test successful batch prediction."""
        mock_predictors.__getitem__.return_value = mock_predictor
        mock_predictors.__contains__.return_value = True

        request_data = {
            "contract_type": "CALL",
            "features": [
                {
                    "dte": 5,
                    "moneyness": 0.99,
                    "mark": 10.5,
                    "strike": 6000.0,
                    "underlying_price": 6060.0,
                    "vix9d": 15.0,
                    "vvix": 90.0,
                    "skew": 140.0,
                },
                {
                    "dte": 10,
                    "moneyness": 0.95,
                    "mark": 15.0,
                    "strike": 6000.0,
                    "underlying_price": 6300.0,
                    "vix9d": 18.0,
                    "vvix": 95.0,
                    "skew": 145.0,
                },
                {
                    "dte": 15,
                    "moneyness": 1.02,
                    "mark": 8.0,
                    "strike": 6000.0,
                    "underlying_price": 5880.0,
                    "vix9d": 12.0,
                    "vvix": 85.0,
                    "skew": 135.0,
                },
            ],
            "version": "latest",
        }

        response = client.post("/predict_deltas", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert "contract_type" in data
        assert len(data["predictions"]) == 3
        assert data["count"] == 3
        assert data["contract_type"] == "CALL"

    def test_empty_batch_prediction(self, client):
        """Test batch prediction with empty list."""
        request_data = {"contract_type": "CALL", "features": []}

        response = client.post("/predict_deltas", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_batch_too_large(self, client):
        """Test batch prediction exceeding limit."""
        # Create a batch with 1001 items (exceeds limit of 1000)
        features = [
            {
                "dte": 5,
                "moneyness": 0.99,
                "mark": 10.5,
                "strike": 6000.0,
                "underlying_price": 6060.0,
                "vix9d": 15.0,
                "vvix": 90.0,
                "skew": 140.0,
            }
        ] * 1001

        request_data = {"contract_type": "CALL", "features": features}

        response = client.post("/predict_deltas", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_contract_type(self, client):
        """Test batch prediction with invalid contract type."""
        request_data = {
            "contract_type": "INVALID",
            "features": [
                {
                    "dte": 5,
                    "moneyness": 0.99,
                    "mark": 10.5,
                    "strike": 6000.0,
                    "underlying_price": 6060.0,
                    "vix9d": 15.0,
                    "vvix": 90.0,
                    "skew": 140.0,
                }
            ],
        }

        response = client.post("/predict_deltas", json=request_data)
        assert response.status_code == 422  # Validation error


class TestModelsEndpoint:
    """Test models listing endpoint."""

    @patch("src.api.app.list_model_versions")
    def test_list_models(self, mock_list_versions, client):
        """Test listing available models."""
        mock_list_versions.side_effect = lambda ct: (
            ["1.0.0", "1.1.0"] if ct == "CALL" else ["1.0.0"]
        )

        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "CALL" in data["models"]
        assert "PUT" in data["models"]


class TestModelInfoEndpoint:
    """Test model info endpoint."""

    @patch("src.api.app.load_model")
    def test_get_model_info(self, mock_load_model, client):
        """Test getting model metadata."""
        mock_metadata = {
            "contract_type": "CALL",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00",
            "model_type": "HistGradientBoostingRegressor",
            "metrics": {"test_mae": 0.05},
        }
        mock_load_model.return_value = (None, None, mock_metadata)

        response = client.get("/models/CALL/1.0.0")
        assert response.status_code == 200
        data = response.json()
        assert data["contract_type"] == "CALL"
        assert data["version"] == "1.0.0"
        assert "metrics" in data

    @patch("src.api.app.load_model")
    def test_model_not_found(self, mock_load_model, client):
        """Test getting info for non-existent model."""
        mock_load_model.side_effect = FileNotFoundError("Model not found")

        response = client.get("/models/CALL/99.0.0")
        assert response.status_code == 404

    def test_invalid_contract_type_in_url(self, client):
        """Test getting info with invalid contract type."""
        response = client.get("/models/INVALID/1.0.0")
        assert response.status_code == 400
