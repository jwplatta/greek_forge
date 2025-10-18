"""FastAPI application for serving option delta predictions."""

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from src.api.predictor import Predictor
from src.utils.constants import (
    VALID_CONTRACT_TYPES,
)
from src.utils.logger import get_logger
from src.utils.model_io import get_latest_version, list_model_versions, load_model

logger = get_logger()

predictors: Dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("Starting Greek Forge API...")

    for contract_type in VALID_CONTRACT_TYPES:
        try:
            version = get_latest_version(contract_type)
            if version:
                predictor = Predictor.load(contract_type, version)
                key = f"{contract_type}:latest"
                predictors[key] = predictor
                logger.info(f"Loaded {contract_type} model v{version}")
            else:
                logger.warning(f"No {contract_type} models found")
        except Exception as e:
            logger.error(f"Failed to load {contract_type} model: {e}")

    logger.info(f"API ready with {len(predictors)} models loaded")

    yield

    logger.info("Shutting down Greek Forge API...")
    predictors.clear()


app = FastAPI(
    title="Greek Forge API",
    description="API for predicting option Greeks",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Greek Forge API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Health check",
)
async def health_check():
    """
    Check API health and loaded models.

    Returns service status and information about loaded models.
    """
    models_loaded = {}
    for key in predictors:
        contract_type, version_key = key.split(":")
        if contract_type not in models_loaded:
            models_loaded[contract_type] = []
        models_loaded[contract_type].append(predictors[key].version)

    return HealthResponse(status="healthy", models_loaded=models_loaded)


@app.get(
    "/models",
    response_model=ModelListResponse,
    tags=["Models"],
    summary="List available models",
)
async def list_models():
    """
    List all available trained models.

    Returns models grouped by contract type with their versions.
    """
    models = {}
    for contract_type in VALID_CONTRACT_TYPES:
        versions = list_model_versions(contract_type)
        models[contract_type] = versions

    return ModelListResponse(models=models)


@app.get(
    "/models/{contract_type}/{version}",
    response_model=ModelInfo,
    tags=["Models"],
    summary="Get model metadata",
    responses={404: {"model": ErrorResponse}},
)
async def get_model_info(contract_type: str, version: str):
    """
    Get metadata for a specific model version.

    Args:
        contract_type: Type of option contract (CALL or PUT)
        version: Model version

    Returns:
        Model metadata including metrics and creation date
    """
    contract_type = contract_type.upper()
    if contract_type not in VALID_CONTRACT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid contract_type: {contract_type}. Must be CALL or PUT",
        )

    try:
        _, _, metadata = load_model(contract_type, version)
        return ModelInfo(
            contract_type=metadata.get("contract_type", contract_type),
            version=metadata.get("version", version),
            created_at=metadata.get("created_at", "unknown"),
            model_type=metadata.get("model_type", "unknown"),
            metrics=metadata.get("metrics", {}),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {contract_type} v{version}",
        )
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model information",
        )


def _get_predictor(contract_type: str, version: str) -> Predictor:
    """
    Get or load a predictor for the given contract type and version.

    Args:
        contract_type: Type of option contract (CALL or PUT)
        version: Model version ('latest' or specific version)

    Returns:
        Loaded predictor instance

    Raises:
        HTTPException: If model cannot be found or loaded
    """
    version_key = version if version != "latest" else None
    cache_key = f"{contract_type}:{version_key or 'latest'}"

    if cache_key not in predictors:
        try:
            if version_key is None:
                version_key = get_latest_version(contract_type)
                if version_key is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No models found for {contract_type}",
                    )

            predictor = Predictor.load(contract_type, version_key)
            predictors[cache_key] = predictor
            logger.info(f"Loaded {contract_type} model v{version_key}")
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {contract_type} v{version_key}",
            )
        except Exception as e:
            logger.error(f"Error loading predictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model",
            )
    else:
        predictor = predictors[cache_key]

    return predictor


@app.post(
    "/predict_delta",
    response_model=SinglePredictionResponse,
    tags=["Delta Predictions"],
    summary="Predict option delta (single)",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_delta(request: SinglePredictionRequest):
    """
    Predict option delta for a single option.

    Args:
        request: Prediction request with contract type, features, and optional version

    Returns:
        Single prediction response with delta value
    """
    try:
        predictor = _get_predictor(request.contract_type, request.version)
        delta = predictor.predict_single(request.features.model_dump())

        return SinglePredictionResponse(
            delta=delta,
            contract_type=request.contract_type,
            model_version=predictor.version,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {e}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )


@app.post(
    "/predict_deltas",
    response_model=BatchPredictionResponse,
    tags=["Delta Predictions"],
    summary="Predict option deltas (batch)",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_deltas(request: BatchPredictionRequest):
    """
    Predict option deltas for multiple options in a single request.

    Batch predictions are more efficient than multiple single predictions.
    Limited to 1000 items per request.

    Args:
        request: Batch prediction request with contract type, list of features, and optional version

    Returns:
        Batch prediction response with list of delta values
    """
    try:
        predictor = _get_predictor(request.contract_type, request.version)
        features_list = [f.model_dump() for f in request.features]
        deltas = predictor.predict_batch(features_list)

        return BatchPredictionResponse(
            predictions=deltas.tolist(),
            contract_type=request.contract_type,
            model_version=predictor.version,
            count=len(deltas),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {e}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
