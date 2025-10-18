"""Curve fitting utilities for smoothing and interpolating delta predictions."""

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit

from src.utils.logger import get_logger

logger = get_logger()


def logistic_function(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """
    Logistic function (sigmoid) for curve fitting.

    Args:
        x: Input array (strikes)
        L: Curve's maximum value
        k: Steepness of the curve
        x0: x-value of the sigmoid's midpoint

    Returns:
        Array of logistic function values
    """
    return L / (1 + np.exp(-k * (x - x0)))


def fit_delta_curve(
    strikes: np.ndarray,
    deltas: np.ndarray,
    steepness_factor: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Fit a logistic curve to predicted deltas.

    Args:
        strikes: Array of strike prices
        deltas: Array of predicted delta values
        steepness_factor: Factor to adjust curve steepness (default: 0.9)

    Returns:
        Tuple of (L, k, x0) optimized parameters for logistic function

    Raises:
        RuntimeError: If curve fitting fails
    """
    p0 = [
        np.max(deltas),  # L: maximum value
        1.0,  # k: steepness
        np.median(strikes),  # x0: midpoint
    ]

    try:
        popt, _ = curve_fit(logistic_function, strikes, deltas, p0=p0)

        L, k, x0 = popt
        k = k * steepness_factor

        logger.info(
            f"Fitted delta curve: L={L:.4f}, k={k:.4f}, x0={x0:.2f} "
            f"(steepness_factor={steepness_factor})"
        )

        return L, k, x0

    except Exception as e:
        logger.error(f"Failed to fit delta curve: {e}")
        raise RuntimeError(f"Curve fitting failed: {e}")


def interpolate_deltas(
    strikes: np.ndarray,
    deltas: np.ndarray,
    strike_min: float,
    strike_max: float,
    strike_step: float = 5.0,
    steepness_factor: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate delta values across a range of strikes using logistic curve fitting.

    Args:
        strikes: Array of strike prices with predictions
        deltas: Array of predicted delta values
        strike_min: Minimum strike for interpolation range
        strike_max: Maximum strike for interpolation range
        strike_step: Step size between strikes (default: 5.0)
        steepness_factor: Factor to adjust curve steepness (default: 0.9)

    Returns:
        Tuple of (interpolated_strikes, interpolated_deltas)

    Example:
        >>> strikes = np.array([6000, 6010, 6020])
        >>> deltas = np.array([0.75, 0.65, 0.55])
        >>> new_strikes, new_deltas = interpolate_deltas(
        ...     strikes, deltas, 5990, 6030, strike_step=10
        ... )
    """
    L, k, x0 = fit_delta_curve(strikes, deltas, steepness_factor)

    interpolated_strikes = np.arange(strike_min, strike_max + strike_step, strike_step)
    interpolated_deltas = logistic_function(interpolated_strikes, L, k, x0)

    logger.info(
        f"Interpolated {len(interpolated_strikes)} strikes from "
        f"{strike_min} to {strike_max} (step={strike_step})"
    )

    return interpolated_strikes, interpolated_deltas


def smooth_deltas(
    strikes: np.ndarray,
    deltas: np.ndarray,
    steepness_factor: float = 0.9,
) -> np.ndarray:
    """
    Smooth predicted deltas using logistic curve fitting.

    This applies the fitted logistic curve back to the original strikes,
    effectively smoothing out noise in the predictions while maintaining
    the same strikes.

    Args:
        strikes: Array of strike prices
        deltas: Array of predicted delta values
        steepness_factor: Factor to adjust curve steepness (default: 0.9)

    Returns:
        Array of smoothed delta values (same length as input)

    Example:
        >>> strikes = np.array([6000, 6010, 6020])
        >>> deltas = np.array([0.751, 0.648, 0.553])  # Noisy predictions
        >>> smoothed = smooth_deltas(strikes, deltas)
        >>> # Returns smooth curve: [0.750, 0.650, 0.550]
    """
    L, k, x0 = fit_delta_curve(strikes, deltas, steepness_factor)

    smoothed_deltas = logistic_function(strikes, L, k, x0)

    logger.info(f"Smoothed {len(deltas)} delta predictions")

    return smoothed_deltas
