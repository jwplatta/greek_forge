# Greek Forge

A machine learning project for predicting option Greeks, specifically delta values for out-of-the-money (OTM) call and put options.

## Overview

Greek Forge uses gradient boosting algorithms to predict option Greeks based on historical data stored in PostgreSQL. The project provides a FastAPI REST API for serving predictions and is containerized with Docker for easy deployment.

## Tech Stack

- **Python 3.10+** - Core language
- **uv** - Fast Python package manager
- **scikit-learn** - Machine learning (HistGradientBoostingRegressor)
- **Optuna** - Hyperparameter optimization
- **PostgreSQL** - Data storage
- **FastAPI** - REST API
- **Docker** - Containerization

## Project Structure

```
greek_forge/
├── doc/                    # Documentation
├── src/                    # Source code
│   ├── data/              # Data preprocessing and loading
│   ├── models/            # Model training and evaluation
│   ├── api/               # FastAPI REST API
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for exploration
├── data/                  # Data files (gitignored)
├── models/                # Saved models (gitignored)
├── pyproject.toml         # Project metadata and dependencies
└── Dockerfile            # Docker configuration
```

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd greek_forge
```

2. Install dependencies using uv:
```bash
# Install core dependencies
uv sync

# Install with dev dependencies (pytest, jupyter)
uv sync --extra dev
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials and configuration
```

### Running Commands

Use `uv run` to execute commands in the project environment:
```bash
# Run Python scripts
uv run python src/models/train.py

# Run tests
uv run pytest

# Start Jupyter
uv run jupyter notebook
```

Alternatively, use the provided Makefile for common tasks:
```bash
# View all available commands
make help

# Code quality
make test          # Run all unit tests
make lint          # Run linting with auto-fix
make format        # Format code with ruff

# Model training
make build-call-model   # Train CALL option model
make build-put-model    # Train PUT option model
make build-models       # Train both models

# Utilities
make clean         # Remove cache files
make sync          # Sync dependencies
```

## Development

See `doc/project_plan.md` for detailed project planning and development workflow.

### Training Models

Train models using the command-line interface:

```bash
# Train CALL option model (default)
uv run python src/utils/model_io.py --contract-type CALL

# Train PUT option model
uv run python src/utils/model_io.py --contract-type PUT

# Specify version and CV folds
uv run python src/utils/model_io.py --contract-type CALL --version 2.0.0 --cv-folds 10
```

Or use the Makefile shortcuts:
```bash
make build-call-model
make build-put-model
make build-models     # Build both
```

### Using Trained Models

```python
from src.api.predictor import Predictor

# Load latest CALL model
predictor = Predictor.load("CALL")

# Or load specific version
predictor = Predictor.load("CALL", version="1.0.0")

# Make predictions
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
print(f"Predicted delta: {delta:.4f}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
