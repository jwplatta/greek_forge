# Greek Forge API Documentation

FastAPI-based REST API for serving option delta predictions.

## Starting the Server

### Development Mode (with auto-reload)
```bash
make serve-dev
# or
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode
```bash
make serve
# or
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to explore all endpoints, see request/response schemas, and test API calls directly from your browser.

## API Endpoints

### Root
**GET /**

Returns basic API information.

**Response:**
```json
{
  "name": "Greek Forge API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### Health Check
**GET /health**

Check API health and loaded models.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "CALL": ["1.0.0"],
    "PUT": ["1.0.0"]
  }
}
```

### Predict Delta (Single)
**POST /predict_delta**

Predict option delta for a single option.

**Request Body:**
```json
{
  "contract_type": "CALL",
  "features": {
    "dte": 5,
    "moneyness": 0.99,
    "mark": 10.5,
    "strike": 6000.0,
    "underlying_price": 6060.0,
    "vix9d": 15.0,
    "vvix": 90.0,
    "skew": 140.0
  },
  "version": "latest"
}
```

**Response:**
```json
{
  "delta": 0.6542,
  "contract_type": "CALL",
  "model_version": "1.0.0"
}
```

**Parameters:**
- `contract_type`: `"CALL"` or `"PUT"`
- `features`: Feature set for prediction
- `version`: Model version to use (default: `"latest"`)

### Predict Deltas (Batch)
**POST /predict_deltas**

Predict option deltas for multiple options in a single request. Batch predictions are more efficient than multiple single predictions.

**Request Body:**
```json
{
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
      "skew": 140.0
    },
    {
      "dte": 10,
      "moneyness": 0.95,
      "mark": 15.0,
      "strike": 6000.0,
      "underlying_price": 6300.0,
      "vix9d": 18.0,
      "vvix": 95.0,
      "skew": 145.0
    }
  ],
  "version": "latest"
}
```

**Response:**
```json
{
  "predictions": [0.6542, 0.7234],
  "contract_type": "CALL",
  "model_version": "1.0.0",
  "count": 2
}
```

**Parameters:**
- `contract_type`: `"CALL"` or `"PUT"`
- `features`: List of feature sets (1-1000 items)
- `version`: Model version to use (default: `"latest"`)

**Feature Fields:**
- `dte` (int): Days to expiration (>= 0)
- `moneyness` (float): Strike / Underlying price ratio (> 0)
- `mark` (float): Option mark price (>= 0)
- `strike` (float): Strike price (> 0)
- `underlying_price` (float): Underlying asset price (> 0)
- `vix9d` (float): 9-day VIX (>= 0)
- `vvix` (float): VVIX volatility index (>= 0)
- `skew` (float): Volatility skew

**Validation:**
- All numeric fields must satisfy the constraints above
- Batch predictions are limited to 1000 items per request
- Empty batch requests are rejected

**Note:** The endpoint naming convention (`predict_delta` vs `predict_deltas`) is designed to accommodate future Greek predictions (e.g., `predict_gamma`, `predict_vega`, `predict_theta`).

### List Models
**GET /models**

List all available trained models.

**Response:**
```json
{
  "models": {
    "CALL": ["1.0.0", "1.1.0"],
    "PUT": ["1.0.0"]
  }
}
```

### Get Model Info
**GET /models/{contract_type}/{version}**

Get metadata for a specific model version.

**Example:** `GET /models/CALL/1.0.0`

**Response:**
```json
{
  "contract_type": "CALL",
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00",
  "model_type": "HistGradientBoostingRegressor",
  "metrics": {
    "test_mae": 0.0234,
    "test_rmse": 0.0356,
    "test_r2": 0.9567
  }
}
```

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message"
}
```

**Example Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "features", "dte"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Python Client Example

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Single prediction
single_request = {
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
    }
}

response = requests.post(f"{BASE_URL}/predict_delta", json=single_request)
result = response.json()
print(f"Predicted delta: {result['delta']:.4f}")

# Batch prediction
batch_request = {
    "contract_type": "PUT",
    "features": [
        {"dte": 5, "moneyness": 1.01, "mark": 8.5, "strike": 6000.0,
         "underlying_price": 5940.0, "vix9d": 16.0, "vvix": 92.0, "skew": 138.0},
        {"dte": 10, "moneyness": 1.05, "mark": 12.0, "strike": 6000.0,
         "underlying_price": 5714.0, "vix9d": 20.0, "vvix": 98.0, "skew": 150.0},
    ]
}

response = requests.post(f"{BASE_URL}/predict_deltas", json=batch_request)
result = response.json()
print(f"Predicted deltas: {result['predictions']}")
print(f"Count: {result['count']}")
```

## Model Loading

The API automatically loads the latest version of CALL and PUT models on startup. Models are cached in memory for fast predictions.

### Model Cache

- Models are loaded on first use and cached
- Cache key format: `"{contract_type}:{version}"`
- Example: `"CALL:latest"`, `"PUT:1.0.0"`

### Loading Additional Models

If you request a specific model version that isn't loaded, the API will:
1. Load the model from disk
2. Cache it for future requests
3. Use it for predictions

This happens transparently - you don't need to manually manage the cache.

## Performance

## Security Considerations

**Current Implementation:**
- No authentication/authorization
- Suitable for internal use only

**For Production:**
- Add API key authentication
- Implement rate limiting
- Use HTTPS (TLS/SSL)
- Add request logging and monitoring
- Consider using API gateway (e.g., Kong, Traefik)

## Docker Deployment

See `Dockerfile` for containerization configuration.

```bash
# Build image
docker build -t greek-forge-api .

# Run container
docker run -p 8000:8000 greek-forge-api
```
