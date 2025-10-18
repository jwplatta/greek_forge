"""Tests for curve fitting utilities."""

import numpy as np
import pytest

from src.utils.curve_fitting import (
    fit_delta_curve,
    interpolate_deltas,
    logistic_function,
    smooth_deltas,
)


class TestLogisticFunction:
    """Test logistic function."""

    def test_logistic_function_midpoint(self):
        """Test that logistic function returns L/2 at midpoint."""
        L, k, x0 = 1.0, 1.0, 100.0
        result = logistic_function(np.array([x0]), L, k, x0)
        assert np.isclose(result[0], L / 2, atol=0.01)

    def test_logistic_function_array(self):
        """Test that logistic function handles arrays."""
        L, k, x0 = 1.0, 1.0, 100.0
        x = np.array([90.0, 100.0, 110.0])
        result = logistic_function(x, L, k, x0)
        assert len(result) == 3
        assert all(0 <= v <= L for v in result)


class TestFitDeltaCurve:
    """Test delta curve fitting."""

    def test_fit_delta_curve_returns_params(self):
        """Test that fit_delta_curve returns three parameters."""
        strikes = np.array([6000.0, 6010.0, 6020.0, 6030.0, 6040.0])
        deltas = np.array([0.9, 0.8, 0.6, 0.4, 0.2])

        L, k, x0 = fit_delta_curve(strikes, deltas)

        assert isinstance(L, float)
        assert isinstance(k, float)
        assert isinstance(x0, float)
        assert L > 0
        assert x0 > 0

    def test_fit_delta_curve_reasonable_fit(self):
        """Test that fitted curve reasonably approximates input."""
        strikes = np.array([6000.0, 6010.0, 6020.0, 6030.0, 6040.0])
        deltas = np.array([0.9, 0.8, 0.6, 0.4, 0.2])

        L, k, x0 = fit_delta_curve(strikes, deltas)
        fitted = logistic_function(strikes, L, k, x0)

        # Check that fitted values are reasonably close to actual
        mae = np.mean(np.abs(fitted - deltas))
        assert mae < 0.1  # Mean absolute error less than 0.1


class TestSmoothDeltas:
    """Test delta smoothing."""

    def test_smooth_deltas_preserves_length(self):
        """Test that smoothing preserves array length."""
        strikes = np.array([6000.0, 6010.0, 6020.0, 6030.0, 6040.0])
        deltas = np.array([0.91, 0.79, 0.62, 0.38, 0.21])  # Slightly noisy

        smoothed = smooth_deltas(strikes, deltas)

        assert len(smoothed) == len(deltas)

    def test_smooth_deltas_reduces_noise(self):
        """Test that smoothing reduces variance."""
        strikes = np.array([6000.0, 6010.0, 6020.0, 6030.0, 6040.0])
        # Add noise to monotonic sequence
        deltas = np.array([0.90, 0.82, 0.59, 0.42, 0.19])

        smoothed = smooth_deltas(strikes, deltas)

        # Smoothed values should be more monotonic
        # Check that differences are more consistent
        original_diffs = np.diff(deltas)
        smoothed_diffs = np.diff(smoothed)

        assert np.std(smoothed_diffs) < np.std(original_diffs)


class TestInterpolateDeltas:
    """Test delta interpolation."""

    def test_interpolate_deltas_expands_range(self):
        """Test that interpolation creates more strikes."""
        strikes = np.array([6000.0, 6020.0, 6040.0])
        deltas = np.array([0.9, 0.6, 0.2])

        new_strikes, new_deltas = interpolate_deltas(
            strikes, deltas, strike_min=5990.0, strike_max=6050.0, strike_step=10.0
        )

        assert len(new_strikes) > len(strikes)
        assert new_strikes[0] == 5990.0
        assert new_strikes[-1] == 6050.0

    def test_interpolate_deltas_fills_gaps(self):
        """Test that interpolation fills gaps between strikes."""
        strikes = np.array([6000.0, 6050.0])  # 50-point gap
        deltas = np.array([0.9, 0.2])

        new_strikes, new_deltas = interpolate_deltas(
            strikes, deltas, strike_min=6000.0, strike_max=6050.0, strike_step=10.0
        )

        # Should have 6000, 6010, 6020, 6030, 6040, 6050
        assert len(new_strikes) == 6
        assert np.allclose(new_strikes, [6000, 6010, 6020, 6030, 6040, 6050])

    def test_interpolate_deltas_values_reasonable(self):
        """Test that interpolated values are reasonable."""
        strikes = np.array([6000.0, 6020.0, 6040.0])
        deltas = np.array([0.9, 0.6, 0.2])

        new_strikes, new_deltas = interpolate_deltas(
            strikes, deltas, strike_min=6000.0, strike_max=6040.0, strike_step=5.0
        )

        # All deltas should be between min and max of original deltas
        assert all(0.0 <= d <= 1.0 for d in new_deltas)
        # Should be roughly monotonic (decreasing for calls)
        assert new_deltas[0] > new_deltas[-1]

    def test_interpolate_with_custom_steepness(self):
        """Test that steepness_factor affects curve."""
        strikes = np.array([6000.0, 6020.0, 6040.0])
        deltas = np.array([0.9, 0.6, 0.2])

        _, deltas_steep = interpolate_deltas(
            strikes,
            deltas,
            strike_min=6000.0,
            strike_max=6040.0,
            strike_step=10.0,
            steepness_factor=1.2,  # Steeper
        )

        _, deltas_gentle = interpolate_deltas(
            strikes,
            deltas,
            strike_min=6000.0,
            strike_max=6040.0,
            strike_step=10.0,
            steepness_factor=0.6,  # Gentler
        )

        # Curves should be different
        assert not np.allclose(deltas_steep, deltas_gentle)
