# Greek Forge

A machine learning project for predicting option Greeks, specifically delta values for out-of-the-money (OTM) call and put options.

## Overview

Greek Forge uses gradient boosting algorithms to predict option Greeks based on historical data stored in PostgreSQL. The project provides a Flask API for serving predictions and is containerized with Docker for easy deployment.

## Tech Stack

- **Python 3.10+** - Core language
- **uv** - Fast Python package manager
- **scikit-learn** - Machine learning (HistGradientBoostingRegressor)
- **Optuna** - Hyperparameter optimization
- **PostgreSQL** - Data storage
- **Flask** - REST API
- **Docker** - Containerization

## Project Structure

```
greek_forge/
├── doc/                    # Documentation
├── src/                    # Source code
│   ├── data/              # Data preprocessing and loading
│   ├── models/            # Model training and evaluation
│   ├── api/               # Flask API
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

## Development

See `doc/project_plan.md` for detailed project planning and development workflow.

## License

TBD
