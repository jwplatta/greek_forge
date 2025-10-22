"""
Delta validation and adjustment utilities.

This module provides functions to ensure predicted option deltas meet
theoretical constraints:
- PUT deltas: -1.0 ≤ delta ≤ 0.0
- CALL deltas: 0.0 ≤ delta ≤ 1.0
- Monotonicity: deltas should be monotonically increasing with strike prices
"""

from typing import List, Tuple
from src.utils.constants import ContractType, CONTRACT_TYPE_CALL, CONTRACT_TYPE_PUT
from src.utils.logger import get_logger

logger = get_logger()


def apply_bounds(deltas: List[float], contract_type: ContractType) -> List[float]:
    """
    Apply theoretical bounds to delta predictions.

    PUT deltas must be in [-1.0, 0.0]
    CALL deltas must be in [0.0, 1.0]

    Args:
        deltas: List of predicted delta values
        contract_type: Either ContractType.CALL or ContractType.PUT

    Returns:
        List of bounded delta values
    """
    bounded_deltas = []
    violations = 0

    if contract_type == CONTRACT_TYPE_PUT:
        for delta in deltas:
            if delta < -1.0:
                bounded_deltas.append(-1.0)
                violations += 1
            elif delta > 0.0:
                bounded_deltas.append(0.0)
                violations += 1
            else:
                bounded_deltas.append(delta)
    else:  # CALL
        for delta in deltas:
            if delta < 0.0:
                bounded_deltas.append(0.0)
                violations += 1
            elif delta > 1.0:
                bounded_deltas.append(1.0)
                violations += 1
            else:
                bounded_deltas.append(delta)

    if violations > 0:
        logger.warning(
            f"Applied bounds to {violations}/{len(deltas)} {contract_type} deltas"
        )

    return bounded_deltas


def enforce_monotonicity(
    deltas: List[float],
    strikes: List[float],
    contract_type: ContractType
) -> List[float]:
    """
    Enforce monotonicity constraints on delta predictions.

    For a given underlying price:
    - CALL deltas should be monotonically decreasing as strike increases
    - PUT deltas should be monotonically increasing as strike increases
      (becoming less negative, i.e., -0.9 to -0.1)

    Non-monotonic values are adjusted by interpolating between neighboring
    monotonic values.

    Args:
        deltas: List of delta values (must already be bounds-checked)
        strikes: List of corresponding strike prices (must be sorted)
        contract_type: Either ContractType.CALL or ContractType.PUT

    Returns:
        List of monotonicity-enforced delta values
    """
    if len(deltas) != len(strikes):
        raise ValueError(
            f"Length mismatch: {len(deltas)} deltas vs {len(strikes)} strikes"
        )

    if len(deltas) <= 1:
        return deltas

    # Check if strikes are sorted
    if not all(strikes[i] <= strikes[i+1] for i in range(len(strikes)-1)):
        raise ValueError("Strikes must be sorted in ascending order")

    monotonic_deltas = deltas.copy()

    # Find violations and fix them
    violations_fixed = 0
    i = 0

    while i < len(monotonic_deltas):
        # Find a sequence of non-monotonic values
        violation_start = None
        violation_end = None

        # Check for monotonicity violation
        if i < len(monotonic_deltas) - 1:
            is_violation = _is_monotonicity_violation(
                monotonic_deltas[i],
                monotonic_deltas[i + 1],
                contract_type
            )

            if is_violation:
                violation_start = i
                # Find the end of the violation sequence
                j = i + 1
                while j < len(monotonic_deltas) - 1:
                    if _is_monotonicity_violation(
                        monotonic_deltas[j],
                        monotonic_deltas[j + 1],
                        contract_type
                    ):
                        j += 1
                    else:
                        break
                violation_end = j

                # Fix the violation by interpolating
                left_idx, right_idx = _find_monotonic_neighbors(
                    violation_start,
                    violation_end,
                    monotonic_deltas,
                    contract_type
                )

                # Interpolate between the monotonic neighbors
                num_violations = violation_end - violation_start + 1
                left_value = monotonic_deltas[left_idx] if left_idx is not None else None
                right_value = monotonic_deltas[right_idx] if right_idx is not None else None

                # Perform linear interpolation
                for k in range(violation_start, violation_end + 1):
                    if left_value is not None and right_value is not None:
                        # Interpolate based on position
                        total_steps = right_idx - left_idx
                        current_step = k - left_idx
                        fraction = current_step / total_steps
                        monotonic_deltas[k] = left_value + fraction * (right_value - left_value)
                    elif left_value is not None:
                        # Only have left neighbor, use it
                        monotonic_deltas[k] = left_value
                    elif right_value is not None:
                        # Only have right neighbor, use it
                        monotonic_deltas[k] = right_value

                violations_fixed += num_violations
                i = violation_end + 1
            else:
                i += 1
        else:
            i += 1

    if violations_fixed > 0:
        logger.warning(
            f"Fixed {violations_fixed} monotonicity violations in {contract_type} deltas"
        )

    return monotonic_deltas


def _is_monotonicity_violation(
    delta1: float,
    delta2: float,
    contract_type: ContractType
) -> bool:
    """
    Check if two consecutive deltas violate monotonicity.

    For CALL options: delta should decrease as strike increases (delta1 > delta2)
    For PUT options: delta should increase as strike increases (delta1 < delta2)
                    (e.g., -0.9 < -0.1, becoming less negative)

    Args:
        delta1: Delta at lower strike
        delta2: Delta at higher strike
        contract_type: Either "CALL" or "PUT"

    Returns:
        True if monotonicity is violated
    """
    if contract_type == CONTRACT_TYPE_CALL:
        # CALL deltas should be decreasing (delta1 >= delta2)
        return delta1 < delta2
    else:  # PUT
        # PUT deltas should be increasing (delta1 <= delta2)
        # This means becoming less negative
        return delta1 > delta2


def _find_monotonic_neighbors(
    violation_start: int,
    violation_end: int,
    deltas: List[float],
    contract_type: ContractType
) -> Tuple[int, int]:
    """
    Find the nearest monotonic neighbors on either side of a violation sequence.

    Args:
        violation_start: Index where violation sequence starts
        violation_end: Index where violation sequence ends
        deltas: List of all delta values
        contract_type: Either ContractType.CALL or ContractType.PUT

    Returns:
        Tuple of (left_neighbor_index, right_neighbor_index)
        Either index may be None if no valid neighbor exists
    """
    left_idx = None
    right_idx = None

    # Find left neighbor
    if violation_start > 0:
        left_idx = violation_start - 1

    # Find right neighbor
    if violation_end < len(deltas) - 1:
        right_idx = violation_end + 1

    return left_idx, right_idx


def validate_and_adjust_deltas(
    deltas: List[float],
    strikes: List[float],
    contract_type: ContractType,
    enforce_monotonic: bool = True
) -> List[float]:
    """
    Apply all validation and adjustment steps to delta predictions.

    This is the main entry point for delta validation. It applies:
    1. Bounds checking (PUT: [-1.0, 0.0], CALL: [0.0, 1.0])
    2. Monotonicity enforcement (if enabled)

    Args:
        deltas: List of predicted delta values
        strikes: List of corresponding strike prices (must be sorted)
        contract_type: Either ContractType.CALL or ContractType.PUT
        enforce_monotonic: Whether to enforce monotonicity (default: True)

    Returns:
        List of validated and adjusted delta values
    """
    # Step 1: Apply bounds
    adjusted_deltas = apply_bounds(deltas, contract_type)

    # Step 2: Enforce monotonicity if requested
    if enforce_monotonic:
        adjusted_deltas = enforce_monotonicity(adjusted_deltas, strikes, contract_type)

    return adjusted_deltas
