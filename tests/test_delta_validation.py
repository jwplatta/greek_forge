"""Tests for delta validation and adjustment utilities."""

import pytest
from src.utils.delta_validation import (
    apply_bounds,
    enforce_monotonicity,
    validate_and_adjust_deltas,
)


class TestApplyBounds:
    """Tests for bounds checking functionality."""

    def test_put_deltas_within_bounds(self):
        """PUT deltas within [-1.0, 0.0] should remain unchanged."""
        deltas = [-0.9, -0.5, -0.1, 0.0, -1.0]
        result = apply_bounds(deltas, "PUT")
        assert result == deltas

    def test_put_deltas_below_lower_bound(self):
        """PUT deltas below -1.0 should be clipped to -1.0."""
        deltas = [-1.5, -2.0, -1.1]
        result = apply_bounds(deltas, "PUT")
        assert result == [-1.0, -1.0, -1.0]

    def test_put_deltas_above_upper_bound(self):
        """PUT deltas above 0.0 should be clipped to 0.0."""
        deltas = [0.1, 0.5, 1.0]
        result = apply_bounds(deltas, "PUT")
        assert result == [0.0, 0.0, 0.0]

    def test_put_deltas_mixed_violations(self):
        """PUT deltas with mixed violations should be corrected."""
        deltas = [-1.5, -0.5, 0.2, -0.1, -2.0, 0.0]
        expected = [-1.0, -0.5, 0.0, -0.1, -1.0, 0.0]
        result = apply_bounds(deltas, "PUT")
        assert result == expected

    def test_call_deltas_within_bounds(self):
        """CALL deltas within [0.0, 1.0] should remain unchanged."""
        deltas = [0.0, 0.1, 0.5, 0.9, 1.0]
        result = apply_bounds(deltas, "CALL")
        assert result == deltas

    def test_call_deltas_below_lower_bound(self):
        """CALL deltas below 0.0 should be clipped to 0.0."""
        deltas = [-0.1, -0.5, -1.0]
        result = apply_bounds(deltas, "CALL")
        assert result == [0.0, 0.0, 0.0]

    def test_call_deltas_above_upper_bound(self):
        """CALL deltas above 1.0 should be clipped to 1.0."""
        deltas = [1.1, 1.5, 2.0]
        result = apply_bounds(deltas, "CALL")
        assert result == [1.0, 1.0, 1.0]

    def test_call_deltas_mixed_violations(self):
        """CALL deltas with mixed violations should be corrected."""
        deltas = [-0.1, 0.3, 1.2, 0.5, -0.5, 1.0]
        expected = [0.0, 0.3, 1.0, 0.5, 0.0, 1.0]
        result = apply_bounds(deltas, "CALL")
        assert result == expected


class TestEnforceMonotonicity:
    """Tests for monotonicity enforcement."""

    def test_call_deltas_monotonic_decreasing(self):
        """CALL deltas that are already monotonically decreasing should remain unchanged."""
        deltas = [0.9, 0.7, 0.5, 0.3, 0.1]
        strikes = [100, 105, 110, 115, 120]
        result = enforce_monotonicity(deltas, strikes, "CALL")
        assert result == deltas

    def test_call_deltas_single_violation(self):
        """CALL deltas with a single violation should be interpolated."""
        deltas = [0.9, 0.7, 0.8, 0.3, 0.1]  # 0.8 violates (increases)
        strikes = [100, 105, 110, 115, 120]
        result = enforce_monotonicity(deltas, strikes, "CALL")

        # The violation at index 2 should be interpolated between 0.7 and 0.3
        assert result[0] == 0.9
        assert result[1] == 0.7
        assert result[2] == pytest.approx(0.5, abs=0.01)  # Linear interpolation
        assert result[3] == 0.3
        assert result[4] == 0.1

    def test_call_deltas_multiple_consecutive_violations(self):
        """CALL deltas with multiple consecutive violations should be interpolated."""
        deltas = [0.9, 0.7, 0.75, 0.8, 0.3, 0.1]  # Multiple violations
        strikes = [100, 105, 110, 115, 120, 125]
        result = enforce_monotonicity(deltas, strikes, "CALL")

        # Violations detected: indices 1-3 violate (should decrease monotonically)
        # They get interpolated between 0.9 and 0.3
        assert result[0] == 0.9
        # Linear interpolation from 0.9 to 0.3 across indices 1-3
        assert 0.3 <= result[1] <= 0.9
        assert 0.3 <= result[2] <= 0.9
        assert 0.3 <= result[3] <= 0.9
        assert result[4] == 0.3
        assert result[5] == 0.1

    def test_put_deltas_monotonic_increasing(self):
        """PUT deltas that are already monotonically increasing should remain unchanged."""
        deltas = [-0.9, -0.7, -0.5, -0.3, -0.1]
        strikes = [100, 105, 110, 115, 120]
        result = enforce_monotonicity(deltas, strikes, "PUT")
        assert result == deltas

    def test_put_deltas_single_violation(self):
        """PUT deltas with a single violation should be interpolated."""
        deltas = [-0.9, -0.7, -0.8, -0.3, -0.1]  # -0.8 violates (decreases)
        strikes = [100, 105, 110, 115, 120]
        result = enforce_monotonicity(deltas, strikes, "PUT")

        # The violation at index 2 should be interpolated between -0.7 and -0.3
        assert result[0] == -0.9
        assert result[1] == -0.7
        assert result[2] == pytest.approx(-0.5, abs=0.01)
        assert result[3] == -0.3
        assert result[4] == -0.1

    def test_put_deltas_multiple_consecutive_violations(self):
        """PUT deltas with multiple consecutive violations should be interpolated."""
        deltas = [-0.9, -0.7, -0.75, -0.8, -0.3, -0.1]  # Multiple violations
        strikes = [100, 105, 110, 115, 120, 125]
        result = enforce_monotonicity(deltas, strikes, "PUT")

        # Violations detected: indices 1-3 violate (should increase monotonically)
        # They get interpolated between -0.9 and -0.3
        assert result[0] == -0.9
        assert -0.9 <= result[1] <= -0.3
        assert -0.9 <= result[2] <= -0.3
        assert -0.9 <= result[3] <= -0.3
        assert result[4] == -0.3
        assert result[5] == -0.1

    def test_single_delta_no_change(self):
        """Single delta should remain unchanged (no monotonicity to check)."""
        deltas = [0.5]
        strikes = [100]
        result = enforce_monotonicity(deltas, strikes, "CALL")
        assert result == deltas

    def test_mismatched_lengths_raises_error(self):
        """Mismatched delta and strike lengths should raise ValueError."""
        deltas = [0.9, 0.7, 0.5]
        strikes = [100, 105]
        with pytest.raises(ValueError, match="Length mismatch"):
            enforce_monotonicity(deltas, strikes, "CALL")

    def test_unsorted_strikes_raises_error(self):
        """Unsorted strikes should raise ValueError."""
        deltas = [0.9, 0.7, 0.5]
        strikes = [100, 115, 105]  # Not sorted
        with pytest.raises(ValueError, match="Strikes must be sorted"):
            enforce_monotonicity(deltas, strikes, "CALL")


class TestValidateAndAdjustDeltas:
    """Tests for the complete validation and adjustment pipeline."""

    def test_call_deltas_full_pipeline(self):
        """Test complete validation pipeline for CALL deltas."""
        # Deltas with both bounds violations and monotonicity violations
        deltas = [1.2, 0.7, 0.75, -0.1, 0.2]
        strikes = [100, 105, 110, 115, 120]

        result = validate_and_adjust_deltas(
            deltas, strikes, "CALL", enforce_monotonic=True
        )

        # After bounds: [1.0, 0.7, 0.75, 0.0, 0.2]
        # After monotonicity: violations get interpolated
        assert result[0] == 1.0  # Bounded from 1.2
        # Indices 1-4 violate monotonicity and get interpolated
        assert 0.0 <= result[1] <= 1.0
        assert 0.0 <= result[2] <= 1.0
        assert 0.0 <= result[3] <= 1.0
        assert 0.0 <= result[4] <= 1.0
        # Results should be monotonically decreasing
        assert result[0] >= result[1] >= result[2] >= result[3] >= result[4]

    def test_put_deltas_full_pipeline(self):
        """Test complete validation pipeline for PUT deltas."""
        # Deltas with both bounds violations and monotonicity violations
        deltas = [-1.5, -0.7, -0.8, 0.1, -0.2]
        strikes = [100, 105, 110, 115, 120]

        result = validate_and_adjust_deltas(
            deltas, strikes, "PUT", enforce_monotonic=True
        )

        # After bounds: [-1.0, -0.7, -0.8, 0.0, -0.2]
        # After monotonicity: violations get interpolated
        assert result[0] == -1.0  # Bounded from -1.5
        # Indices 1-4 violate monotonicity and get interpolated
        assert -1.0 <= result[1] <= 0.0
        assert -1.0 <= result[2] <= 0.0
        assert -1.0 <= result[3] <= 0.0
        assert -1.0 <= result[4] <= 0.0
        # Results should be monotonically increasing (less negative)
        assert result[0] <= result[1] <= result[2] <= result[3] <= result[4]

    def test_skip_monotonicity_enforcement(self):
        """Test that monotonicity can be optionally disabled."""
        deltas = [0.9, 0.7, 0.8, 0.3, 0.1]  # Has monotonicity violation
        strikes = [100, 105, 110, 115, 120]

        result = validate_and_adjust_deltas(
            deltas, strikes, "CALL", enforce_monotonic=False
        )

        # Only bounds should be applied (no violations here), so deltas unchanged
        assert result == deltas

    def test_perfect_deltas_no_change(self):
        """Deltas that meet all constraints should remain unchanged."""
        deltas = [0.9, 0.7, 0.5, 0.3, 0.1]
        strikes = [100, 105, 110, 115, 120]

        result = validate_and_adjust_deltas(
            deltas, strikes, "CALL", enforce_monotonic=True
        )

        assert result == deltas

    def test_empty_deltas(self):
        """Empty delta list should be handled gracefully."""
        deltas = []
        strikes = []

        result = validate_and_adjust_deltas(
            deltas, strikes, "CALL", enforce_monotonic=True
        )

        assert result == []


class TestRealWorldScenarios:
    """Tests based on realistic option delta scenarios."""

    def test_atm_call_option_chain(self):
        """Test ATM call option chain with strikes around current price."""
        # Realistic ATM call deltas (decreasing as strike increases)
        deltas = [0.85, 0.70, 0.55, 0.40, 0.25, 0.15]
        strikes = [95, 100, 105, 110, 115, 120]

        result = validate_and_adjust_deltas(
            deltas, strikes, "CALL", enforce_monotonic=True
        )

        # Should remain unchanged as they're already valid
        assert result == deltas

    def test_atm_put_option_chain(self):
        """Test ATM put option chain with strikes around current price."""
        # Realistic ATM put deltas (increasing/less negative as strike increases)
        deltas = [-0.85, -0.70, -0.55, -0.40, -0.25, -0.15]
        strikes = [95, 100, 105, 110, 115, 120]

        result = validate_and_adjust_deltas(
            deltas, strikes, "PUT", enforce_monotonic=True
        )

        # Should remain unchanged as they're already valid
        assert result == deltas

    def test_deep_itm_call_edge_case(self):
        """Test deep ITM calls approaching delta of 1.0."""
        # Some predictions might exceed 1.0 for deep ITM
        deltas = [1.05, 0.98, 0.92, 0.85]
        strikes = [80, 85, 90, 95]

        result = validate_and_adjust_deltas(
            deltas, strikes, "CALL", enforce_monotonic=True
        )

        assert result[0] == 1.0  # Clipped
        assert result[1] == 0.98
        assert result[2] == 0.92
        assert result[3] == 0.85

    def test_deep_otm_put_edge_case(self):
        """Test deep OTM puts approaching delta of 0.0."""
        # Some predictions might go slightly positive
        deltas = [-0.15, -0.08, 0.02, 0.05]
        strikes = [115, 120, 125, 130]

        result = validate_and_adjust_deltas(
            deltas, strikes, "PUT", enforce_monotonic=True
        )

        assert result[0] == -0.15
        assert result[1] == -0.08
        assert result[2] == 0.0  # Clipped
        assert result[3] == 0.0  # Clipped
