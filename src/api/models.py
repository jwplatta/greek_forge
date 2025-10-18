"""Pydantic models for API request/response validation."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.constants import ContractType


class OptionFeatures(BaseModel):
    """Features for a single option prediction."""

    dte: int = Field(..., ge=0, description="Days to expiration")
    moneyness: float = Field(..., gt=0, description="Strike / Underlying price ratio")
    mark: float = Field(..., ge=0, description="Option mark price")
    strike: float = Field(..., gt=0, description="Strike price")
    underlying_price: float = Field(..., gt=0, description="Underlying asset price")
    vix9d: float = Field(..., ge=0, description="9-day VIX")
    vvix: float = Field(..., ge=0, description="VVIX volatility index")
    skew: float = Field(..., description="Volatility skew")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
    )


class SinglePredictionRequest(BaseModel):
    """Request model for single delta prediction."""

    contract_type: ContractType = Field(..., description="Type of option contract")
    features: OptionFeatures = Field(..., description="Feature set for prediction")
    version: str = Field(
        default="latest", description="Model version to use (default: latest)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch delta prediction."""

    contract_type: ContractType = Field(..., description="Type of option contract")
    features: List[OptionFeatures] = Field(
        ..., description="List of feature sets for batch prediction"
    )
    version: str = Field(
        default="latest", description="Model version to use (default: latest)"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Ensure batch predictions don't exceed reasonable limits."""
        if len(v) == 0:
            raise ValueError("Batch prediction requires at least one feature set")
        if len(v) > 1000:
            raise ValueError("Batch prediction limited to 1000 items per request")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                ],
                "version": "latest",
            }
        }
    )


class SinglePredictionResponse(BaseModel):
    """Response for single prediction."""

    delta: float = Field(..., description="Predicted delta value")
    contract_type: str = Field(..., description="Type of option contract")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""

    predictions: List[float] = Field(..., description="List of predicted delta values")
    contract_type: str = Field(..., description="Type of option contract")
    model_version: str = Field(..., description="Model version used")
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: dict = Field(..., description="Loaded models by contract type")


class ModelInfo(BaseModel):
    """Model metadata information."""

    contract_type: str = Field(..., description="Type of option contract")
    version: str = Field(..., description="Model version")
    created_at: str = Field(..., description="Model creation timestamp")
    model_type: str = Field(..., description="Type of ML model")
    metrics: dict = Field(..., description="Model evaluation metrics")


class ModelListResponse(BaseModel):
    """Response for listing available models."""

    models: dict = Field(
        ..., description="Available models grouped by contract type and version"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(..., description="Error message")
