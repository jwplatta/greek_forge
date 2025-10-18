"""Tests for curve fitting utilities."""

import numpy as np

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
        # Use realistic option chain data (more strikes)
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        L, k, x0 = fit_delta_curve(strikes, deltas)

        assert isinstance(L, float)
        assert isinstance(k, float)
        assert isinstance(x0, float)
        assert L > 0
        assert x0 > 0

    def test_fit_delta_curve_reasonable_fit(self):
        """Test that fitted curve reasonably approximates input."""
        # Use realistic option chain data (more strikes)
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        L, k, x0 = fit_delta_curve(strikes, deltas)
        fitted = logistic_function(strikes, L, k, x0)

        # Check that fitted values are reasonably close to actual
        mae = np.mean(np.abs(fitted - deltas))
        assert mae < 0.15  # Mean absolute error less than 0.15


class TestSmoothDeltas:
    """Test delta smoothing."""

    def test_smooth_deltas_preserves_length(self):
        """Test that smoothing preserves array length."""
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        smoothed = smooth_deltas(strikes, deltas)

        assert len(smoothed) == len(deltas)

    def test_smooth_deltas_produces_smooth_curve(self):
        """Test that smoothing produces values within expected range."""
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        smoothed = smooth_deltas(strikes, deltas)

        # All smoothed values should be reasonable (within 0-1 range)
        assert all(smoothed <= 1.0)
        assert all(smoothed >= 0.0)
        # Should be monotonically decreasing (or close to it)
        assert smoothed[0] > smoothed[-1]
        # Should be close to original values
        mae = np.mean(np.abs(smoothed - deltas))
        assert mae < 0.1


class TestInterpolateDeltas:
    """Test delta interpolation."""

    def test_interpolate_deltas_expands_range(self):
        """Test that interpolation creates more strikes."""
        # Use realistic option chain data
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        new_strikes, new_deltas = interpolate_deltas(
            strikes, deltas, strike_min=6000.0, strike_max=6050.0, strike_step=5.0
        )

        assert len(new_strikes) > len(strikes)
        assert new_strikes[0] == 6000.0
        assert new_strikes[-1] == 6050.0

    def test_interpolate_deltas_fills_gaps(self):
        """Test that interpolation fills gaps between strikes."""
        # Use more data points to avoid curve fitting issues
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        new_strikes, new_deltas = interpolate_deltas(
            strikes, deltas, strike_min=6000.0, strike_max=6050.0, strike_step=10.0
        )

        # Should have 6000, 6010, 6020, 6030, 6040, 6050
        assert len(new_strikes) == 6
        assert np.allclose(new_strikes, [6000, 6010, 6020, 6030, 6040, 6050])

    def test_interpolate_deltas_values_reasonable(self):
        """Test that interpolated values are reasonable."""
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        new_strikes, new_deltas = interpolate_deltas(
            strikes, deltas, strike_min=6000.0, strike_max=6045.0, strike_step=5.0
        )

        # All deltas should be between 0 and 1
        assert all(0.0 <= d <= 1.0 for d in new_deltas)
        # Should be roughly monotonic (decreasing for calls)
        assert new_deltas[0] > new_deltas[-1]

    def test_interpolate_with_custom_steepness(self):
        """Test that steepness_factor affects curve."""
        strikes = np.array(
            [6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0]
        )
        deltas = np.array([0.75, 0.72, 0.71, 0.69, 0.67, 0.60, 0.50, 0.40])

        _, deltas_steep = interpolate_deltas(
            strikes,
            deltas,
            strike_min=6005.0,
            strike_max=6040.0,
            strike_step=5.0,
            steepness_factor=1.2,  # Steeper
        )

        _, deltas_gentle = interpolate_deltas(
            strikes,
            deltas,
            strike_min=6005.0,
            strike_max=6040.0,
            strike_step=5.0,
            steepness_factor=0.6,  # Gentler
        )

        # Curves should be different
        assert not np.allclose(deltas_steep, deltas_gentle)
