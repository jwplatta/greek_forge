"""Tests for trainer module."""

import numpy as np

from src.data.loader import fetch_call_samples
from src.data.preprocessor import preprocess_training_data
from src.models.trainer import DeltaPredictor, train_model, train_with_cv


class TestDeltaPredictor:
    """Test DeltaPredictor class."""

    def test_predictor_initialization_with_defaults(self):
        """Test that predictor initializes with default parameters."""
        predictor = DeltaPredictor()

        assert predictor.params["max_iter"] == 125
        assert predictor.params["learning_rate"] == 0.15
        assert predictor.params["max_depth"] == 10

    def test_predictor_initialization_with_custom_params(self):
        """Test that predictor accepts custom parameters."""
        custom_params = {"max_iter": 50, "learning_rate": 0.1}

        predictor = DeltaPredictor(**custom_params)

        assert predictor.params["max_iter"] == 50
        assert predictor.params["learning_rate"] == 0.1

    def test_predictor_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        df = fetch_call_samples(limit=50)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        predictor = DeltaPredictor()
        result = predictor.fit(X_train, y_train)

        assert result is predictor

    def test_predictor_can_make_predictions(self):
        """Test that predictor can make predictions after fitting."""
        df = fetch_call_samples(limit=50)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        predictor = DeltaPredictor()
        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)

    def test_predictor_predictions_are_numeric(self):
        """Test that predictions are numeric values."""
        df = fetch_call_samples(limit=50)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        predictor = DeltaPredictor()
        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)

        assert np.issubdtype(predictions.dtype, np.number)
        assert not np.isnan(predictions).any()

    def test_get_params_returns_dict(self):
        """Test that get_params returns a dictionary."""
        predictor = DeltaPredictor()
        params = predictor.get_params()

        assert isinstance(params, dict)
        assert "max_iter" in params
        assert "learning_rate" in params


class TestTrainModel:
    """Test train_model function."""

    def test_train_model_returns_predictor(self):
        """Test that train_model returns a DeltaPredictor."""
        df = fetch_call_samples(limit=50)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        model = train_model(X_train, y_train)

        assert isinstance(model, DeltaPredictor)

    def test_train_model_with_custom_params(self):
        """Test that train_model accepts custom parameters."""
        df = fetch_call_samples(limit=50)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        custom_params = {"max_iter": 50, "learning_rate": 0.1}
        model = train_model(X_train, y_train, params=custom_params)

        assert model.params["max_iter"] == 50
        assert model.params["learning_rate"] == 0.1

    def test_train_model_can_predict(self):
        """Test that trained model can make predictions."""
        df = fetch_call_samples(limit=50)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)


class TestTrainWithCV:
    """Test train_with_cv function."""

    def test_train_with_cv_returns_two_values(self):
        """Test that train_with_cv returns model and cv_results."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        result = train_with_cv(X_train, y_train, cv=3)

        assert len(result) == 2

    def test_train_with_cv_model_type(self):
        """Test that train_with_cv returns a DeltaPredictor."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        model, cv_results = train_with_cv(X_train, y_train, cv=3)

        assert isinstance(model, DeltaPredictor)

    def test_train_with_cv_results_structure(self):
        """Test that cv_results has expected structure."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        model, cv_results = train_with_cv(X_train, y_train, cv=3)

        assert "scores" in cv_results
        assert "mean" in cv_results
        assert "std" in cv_results
        assert "metric" in cv_results

    def test_train_with_cv_scores_length(self):
        """Test that cv_results has correct number of scores."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        cv_folds = 3
        model, cv_results = train_with_cv(X_train, y_train, cv=cv_folds)

        assert len(cv_results["scores"]) == cv_folds

    def test_train_with_cv_model_can_predict(self):
        """Test that model from train_with_cv can make predictions."""
        df = fetch_call_samples(limit=100)
        X_train, X_test, y_train, y_test, _ = preprocess_training_data(
            df, balance_samples=False
        )

        model, cv_results = train_with_cv(X_train, y_train, cv=3)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
