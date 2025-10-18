"""Tests for model I/O utilities."""

import pytest
import tempfile
from pathlib import Path
import json

from src.data.loader import fetch_call_samples
from src.data.preprocessor import preprocess_training_data
from src.models.trainer import train_model
from src.utils.constants import CONTRACT_TYPE_CALL, CONTRACT_TYPE_PUT
from src.utils.model_io import (
    get_model_dir,
    save_model,
    load_model,
    list_model_versions,
    get_latest_version,
)


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trained_model():
    """Fixture that provides a trained model and preprocessor."""
    df = fetch_call_samples(limit=100)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(
        df, balance_samples=False
    )
    model = train_model(X_train, y_train)

    return model, preprocessor, X_train


class TestGetModelDir:
    """Test get_model_dir function."""

    def test_get_model_dir_for_calls(self, temp_model_dir):
        """Test that directory path is correct for CALL options."""
        model_dir = get_model_dir(CONTRACT_TYPE_CALL, "1.0.0", base_dir=temp_model_dir)

        assert "calls" in str(model_dir)
        assert "v1.0.0" in str(model_dir)

    def test_get_model_dir_for_puts(self, temp_model_dir):
        """Test that directory path is correct for PUT options."""
        model_dir = get_model_dir(CONTRACT_TYPE_PUT, "1.0.0", base_dir=temp_model_dir)

        assert "puts" in str(model_dir)
        assert "v1.0.0" in str(model_dir)


class TestSaveModel:
    """Test save_model function."""

    def test_save_model_creates_directory(self, temp_model_dir, trained_model):
        """Test that save_model creates the model directory."""
        model, preprocessor, X_train = trained_model

        model_dir = save_model(
            model=model,
            preprocessor=preprocessor,
            contract_type=CONTRACT_TYPE_CALL,
            version="1.0.0",
            metrics={"mae": 0.001},
            hyperparameters={"max_iter": 100},
            base_dir=temp_model_dir,
        )

        assert model_dir.exists()
        assert model_dir.is_dir()

    def test_save_model_creates_all_artifacts(self, temp_model_dir, trained_model):
        """Test that all expected files are created."""
        model, preprocessor, X_train = trained_model

        model_dir = save_model(
            model=model,
            preprocessor=preprocessor,
            contract_type=CONTRACT_TYPE_CALL,
            version="1.0.0",
            metrics={"mae": 0.001},
            hyperparameters={"max_iter": 100},
            feature_names=X_train.columns.tolist(),
            base_dir=temp_model_dir,
        )

        assert (model_dir / "model.joblib").exists()
        assert (model_dir / "preprocessor.joblib").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "feature_names.json").exists()

    def test_save_model_metadata_content(self, temp_model_dir, trained_model):
        """Test that metadata contains expected information."""
        model, preprocessor, X_train = trained_model

        metrics = {"mae": 0.001, "rmse": 0.002}
        hyperparameters = {"max_iter": 100, "learning_rate": 0.1}

        model_dir = save_model(
            model=model,
            preprocessor=preprocessor,
            contract_type=CONTRACT_TYPE_CALL,
            version="1.0.0",
            metrics=metrics,
            hyperparameters=hyperparameters,
            base_dir=temp_model_dir,
        )

        # Load and check metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        assert metadata["contract_type"] == CONTRACT_TYPE_CALL
        assert metadata["version"] == "1.0.0"
        assert metadata["metrics"] == metrics
        assert metadata["hyperparameters"] == hyperparameters
        assert "created_at" in metadata


class TestLoadModel:
    """Test load_model function."""

    def test_load_model_returns_three_values(self, temp_model_dir, trained_model):
        """Test that load_model returns model, preprocessor, and metadata."""
        model, preprocessor, X_train = trained_model

        # Save first
        save_model(
            model=model,
            preprocessor=preprocessor,
            contract_type=CONTRACT_TYPE_CALL,
            version="1.0.0",
            metrics={"mae": 0.001},
            hyperparameters={"max_iter": 100},
            base_dir=temp_model_dir,
        )

        # Load
        result = load_model(CONTRACT_TYPE_CALL, "1.0.0", base_dir=temp_model_dir)

        assert len(result) == 3

    def test_load_model_can_predict(self, temp_model_dir, trained_model):
        """Test that loaded model can make predictions."""
        model, preprocessor, X_train = trained_model

        # Save
        save_model(
            model=model,
            preprocessor=preprocessor,
            contract_type=CONTRACT_TYPE_CALL,
            version="1.0.0",
            metrics={"mae": 0.001},
            hyperparameters={"max_iter": 100},
            base_dir=temp_model_dir,
        )

        # Load
        loaded_model, loaded_preprocessor, metadata = load_model(
            CONTRACT_TYPE_CALL, "1.0.0", base_dir=temp_model_dir
        )

        # Test prediction
        predictions = loaded_model.predict(X_train[:5])

        assert len(predictions) == 5

    def test_load_model_nonexistent_raises_error(self, temp_model_dir):
        """Test that loading nonexistent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model(CONTRACT_TYPE_CALL, "99.99.99", base_dir=temp_model_dir)


class TestListModelVersions:
    """Test list_model_versions function."""

    def test_list_model_versions_empty(self, temp_model_dir):
        """Test that empty directory returns empty list."""
        versions = list_model_versions(CONTRACT_TYPE_CALL, base_dir=temp_model_dir)

        assert versions == []

    def test_list_model_versions_returns_versions(self, temp_model_dir, trained_model):
        """Test that list_model_versions finds saved versions."""
        model, preprocessor, X_train = trained_model

        # Save multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            save_model(
                model=model,
                preprocessor=preprocessor,
                contract_type=CONTRACT_TYPE_CALL,
                version=version,
                metrics={"mae": 0.001},
                hyperparameters={"max_iter": 100},
                base_dir=temp_model_dir,
            )

        versions = list_model_versions(CONTRACT_TYPE_CALL, base_dir=temp_model_dir)

        assert len(versions) == 3
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions

    def test_list_model_versions_sorted(self, temp_model_dir, trained_model):
        """Test that versions are returned in sorted order."""
        model, preprocessor, X_train = trained_model

        # Save in random order
        for version in ["2.0.0", "1.0.0", "1.1.0"]:
            save_model(
                model=model,
                preprocessor=preprocessor,
                contract_type=CONTRACT_TYPE_CALL,
                version=version,
                metrics={"mae": 0.001},
                hyperparameters={"max_iter": 100},
                base_dir=temp_model_dir,
            )

        versions = list_model_versions(CONTRACT_TYPE_CALL, base_dir=temp_model_dir)

        assert versions == ["1.0.0", "1.1.0", "2.0.0"]


class TestGetLatestVersion:
    """Test get_latest_version function."""

    def test_get_latest_version_empty(self, temp_model_dir):
        """Test that get_latest_version returns None for empty directory."""
        latest = get_latest_version(CONTRACT_TYPE_CALL, base_dir=temp_model_dir)

        assert latest is None

    def test_get_latest_version_returns_highest(self, temp_model_dir, trained_model):
        """Test that get_latest_version returns the highest version."""
        model, preprocessor, X_train = trained_model

        # Save multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0", "1.5.0"]:
            save_model(
                model=model,
                preprocessor=preprocessor,
                contract_type=CONTRACT_TYPE_CALL,
                version=version,
                metrics={"mae": 0.001},
                hyperparameters={"max_iter": 100},
                base_dir=temp_model_dir,
            )

        latest = get_latest_version(CONTRACT_TYPE_CALL, base_dir=temp_model_dir)

        assert latest == "2.0.0"
