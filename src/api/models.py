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
                },
                "version": "latest",
            }
        }
    )


class InterpolationOptions(BaseModel):
    """Options for interpolating missing strikes in delta predictions."""

    strike_min: float = Field(..., gt=0, description="Minimum strike for interpolation")
    strike_max: float = Field(..., gt=0, description="Maximum strike for interpolation")
    strike_step: float = Field(
        default=5.0, gt=0, description="Step size between strikes (default: 5.0)"
    )

    @field_validator("strike_max")
    @classmethod
    def validate_strike_range(cls, v, info):
        """Ensure strike_max is greater than strike_min."""
        if "strike_min" in info.data and v <= info.data["strike_min"]:
            raise ValueError("strike_max must be greater than strike_min")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strike_min": 5900.0,
                "strike_max": 6100.0,
                "strike_step": 5.0,
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
    smooth: bool = Field(
        default=False,
        description="Apply logistic curve smoothing to predictions (default: false)",
    )
    interpolate: bool = Field(
        default=False,
        description="Interpolate deltas for missing strikes (default: false)",
    )
    interpolation_options: InterpolationOptions | None = Field(
        default=None,
        description="Required if interpolate=true. Defines strike range and step size.",
    )
    steepness_factor: float = Field(
        default=0.9,
        gt=0,
        le=2.0,
        description="Curve steepness adjustment factor for smoothing/interpolation (default: 0.9)",
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

    @field_validator("interpolation_options")
    @classmethod
    def validate_interpolation_options(cls, v, info):
        """Ensure interpolation_options is provided when interpolate=true."""
        if info.data.get("interpolate") and v is None:
            raise ValueError("interpolation_options is required when interpolate=true")
        if not info.data.get("interpolate") and v is not None:
            raise ValueError(
                "interpolation_options should only be provided when interpolate=true"
            )
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
                    },
                    {
                        "dte": 10,
                        "moneyness": 0.95,
                        "mark": 15.0,
                        "strike": 6000.0,
                        "underlying_price": 6300.0,
                        "vix9d": 18.0,
                        "vvix": 95.0,
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
    strikes: List[float] | None = Field(
        default=None,
        description="List of strike prices (included when interpolation is used)",
    )
    contract_type: str = Field(..., description="Type of option contract")
    model_version: str = Field(..., description="Model version used")
    count: int = Field(..., description="Number of predictions")
    smoothed: bool = Field(default=False, description="Whether smoothing was applied")
    interpolated: bool = Field(
        default=False, description="Whether interpolation was applied"
    )


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
