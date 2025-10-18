# Machine Learning Architecture

## Overview

Greek Forge follows a modular ML architecture that separates data processing, model training, optimization, and serving into distinct components. This structure ensures maintainability, testability, and easy model versioning.

## Project Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── loader.py          # Data fetching from PostgreSQL
│   ├── preprocessor.py    # Feature engineering, encoding, scaling
│   └── option_samples.sql # SQL query for training data
├── models/
│   ├── __init__.py
│   ├── trainer.py         # Model training logic
│   ├── evaluator.py       # Model evaluation metrics
│   └── optimizer.py       # Hyperparameter tuning with Optuna
├── api/
│   ├── __init__.py
│   ├── app.py            # FastAPI application
│   └── predictor.py      # Model loading & prediction service
└── utils/
    ├── __init__.py
    ├── db.py             # Database connection utilities
    └── model_io.py       # Model save/load utilities
```

## Component Responsibilities

### 1. Data Layer (`src/data/`)

#### `loader.py`
- **Purpose**: Fetch raw data from PostgreSQL
- **Key Functions**:
  - `fetch_option_samples()`: Parameterized query for CALL/PUT data
  - `fetch_call_samples()`, `fetch_put_samples()`: Convenience wrappers
  - `balance_delta_samples()`: Address class imbalance via binning
  - `prepare_features_target()`: Initial feature/target separation

#### `preprocessor.py`
- **Purpose**: Transform raw data into ML-ready features
- **Responsibilities**:
  - Label encoding for categorical features (contract_type)
  - Train/test splitting with stratification
  - Feature scaling/normalization (if needed)
  - Fit and persist transformers (encoders, scalers)
- **Key Functions**:
  - `get_preprocessor()`: Returns fitted preprocessing pipeline
  - `preprocess_training_data()`: Full preprocessing for training
  - `preprocess_prediction_data()`: Transform new data for predictions

### 2. Model Layer (`src/models/`)

#### `trainer.py`
- **Purpose**: Model training orchestration
- **Responsibilities**:
  - Initialize HistGradientBoostingRegressor
  - Fit model on preprocessed data
  - Track training metrics
  - Cross-validation
- **Key Functions**:
  - `train_model()`: Train with given hyperparameters
  - `train_with_cv()`: Train with cross-validation

#### `optimizer.py`
- **Purpose**: Hyperparameter optimization using Optuna
- **Responsibilities**:
  - Define hyperparameter search space
  - Run Optuna trials
  - Track best parameters
  - Save optimization history
- **Key Functions**:
  - `optimize_hyperparameters()`: Run Optuna study
  - `get_best_params()`: Retrieve optimal parameters

#### `evaluator.py`
- **Purpose**: Model evaluation and metrics
- **Responsibilities**:
  - Calculate MAE, RMSE, R²
  - Permutation importance
  - Learning curves
  - Validation curves
- **Key Functions**:
  - `evaluate_model()`: Comprehensive evaluation
  - `calculate_feature_importance()`: Permutation importance
  - `plot_learning_curves()`: Training diagnostics

### 3. Utilities Layer (`src/utils/`)

#### `model_io.py`
- **Purpose**: Model persistence and versioning
- **Responsibilities**:
  - Save trained models with metadata
  - Load models for inference
  - Version management
  - Save preprocessing artifacts (encoders, scalers)
- **Key Functions**:
  - `save_model()`: Persist model + transformers + metadata
  - `load_model()`: Load for predictions
  - `list_model_versions()`: View available models

**Model Artifact Structure**:
```
models/
├── calls/
│   ├── v1.0.0/
│   │   ├── model.joblib          # Trained model
│   │   ├── preprocessor.joblib   # Fitted transformers
│   │   ├── metadata.json         # Training date, metrics, params
│   │   └── feature_names.json    # Expected feature columns
│   └── v1.1.0/
│       └── ...
└── puts/
    └── v1.0.0/
        └── ...
```

### 4. API Layer (`src/api/`)

#### `predictor.py`
- **Purpose**: Prediction service (stateless)
- **Responsibilities**:
  - Load model + preprocessor on initialization
  - Transform incoming data
  - Generate predictions
  - Handle errors gracefully
- **Key Functions**:
  - `Predictor.load()`: Load model artifacts
  - `Predictor.predict()`: Generate predictions
  - `Predictor.predict_proba()`: Confidence scores (if applicable)

#### `app.py`
- **Purpose**: FastAPI REST API
- **Endpoints**:
  - `POST /predict`: Single prediction
  - `POST /predict/batch`: Batch predictions
  - `GET /health`: Health check
  - `GET /model/info`: Model metadata

## Workflow

### Training Workflow

```python
# 1. Load data
from src.data.loader import fetch_call_samples

df = fetch_call_samples()

# 2. Preprocess
from src.data.preprocessor import preprocess_training_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_training_data(df)

# 3. Optimize (optional)
from src.models.optimizer import optimize_hyperparameters

best_params = optimize_hyperparameters(X_train, y_train)

# 4. Train
from src.models.trainer import train_model

model = train_model(X_train, y_train, params=best_params)

# 5. Evaluate
from src.models.evaluator import evaluate_model

metrics = evaluate_model(model, X_test, y_test)

# 6. Save
from src.utils.model_io import save_model

save_model(
    model=model,
    preprocessor=preprocessor,
    contract_type='CALL',
    version='1.0.0',
    metrics=metrics,
    hyperparameters=best_params
)
```

### Prediction Workflow

```python
# 1. Initialize predictor (once at startup)
from src.api.predictor import Predictor

predictor = Predictor.load(contract_type='CALL', version='1.0.0')

# 2. Make predictions
input_data = {
    'dte': 5,
    'moneyness': 0.99,
    'mark': 10.5,
    'strike': 6000,
    'underlying_price': 6060,
    'vix9d': 15.0,
    'vvix': 90.0,
    'skew': 140.0
}

delta = predictor.predict(input_data)
```
