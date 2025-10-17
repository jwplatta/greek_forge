# Greek Forge - Project Plan

## Project Overview
Machine learning project to predict option Greeks (delta values) for OTM call and put options.

## Tech Stack
- **Language**: Python 3.x
- **ML Framework**: scikit-learn (HistGradientBoostingRegressor)
- **Optimization**: Optuna
- **Database**: PostgreSQL
- **API**: Flask
- **Containerization**: Docker

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
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md             # Project documentation
```

## Getting Started Tasks
1. Set up project folder structure
2. Initialize git repository with .gitignore
3. Create requirements.txt with core dependencies
4. Create basic README

## Core Dependencies
- scikit-learn
- optuna
- psycopg2-binary
- flask
- pandas
- numpy
- python-dotenv

## Development Workflow
1. Data preprocessing and exploration
2. Feature engineering
3. Model training with hyperparameter tuning
4. Model evaluation
5. API development
6. Docker containerization
7. Testing and deployment
