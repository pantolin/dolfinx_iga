"""Tolerance utilities for floating-point comparisons in IGA applications."""

from typing import TypedDict

import numpy as np


def _check_dtype(dtype: np.dtype | type[np.floating]) -> None:
    """Ensure the dtype is a NumPy dtype. Otherwise, raises an exception.

    Args:
        dtype: Input dtype (can be a NumPy dtype or a floating-point type)
    """
    if dtype not in (np.float16, np.float32, np.float64, np.longdouble):
        raise ValueError(f"Unsupported dtype: {dtype}")


def _get_tolerance(
    dtype: np.dtype | type[np.floating],
    values: list[float],
) -> float:
    """Get the tolerance value for a specific dtype and a list of values.

    Args:
        dtype: NumPy floating-point data type
        values: List of floating-point values

    Returns:
        Tolerance value for the given dtype
    """

    _check_dtype(dtype)

    if dtype == np.float16:
        return values[0]
    elif dtype == np.float32:
        return values[1]
    elif dtype == np.float64:
        return values[2]
    else:  # if dtype == np.longdouble:
        return values[3]


def get_default_tolerance(dtype: np.dtype | type[np.floating]) -> float:
    """Get a reasonable default tolerance for floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type or numpy scalar type

    Returns:
        Recommended tolerance value for the given dtype

    Examples:
        >>> get_default_tolerance(np.float32)
        1e-06
        >>> get_default_tolerance(np.float64)
        1e-12
    """

    return _get_tolerance(
        dtype,
        [float(1e-3), float(1e-6), float(1e-12), float(1e-15)],
    )


def get_strict_tolerance(dtype: np.dtype | type[np.floating]) -> float:
    """Get a strict tolerance for high-precision floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Strict tolerance value for the given dtype (typically for parametric coordinates)
    """

    return _get_tolerance(
        dtype,
        [float(1e-4), float(1e-7), float(1e-15), float(1e-18)],
    )


def get_conservative_tolerance(dtype: np.dtype | type[np.floating]) -> float:
    """Get a conservative tolerance for robust floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Conservative tolerance value for the given dtype
    """

    return _get_tolerance(
        dtype,
        [float(1e-2), float(1e-5), float(1e-10), float(1e-12)],
    )


def get_machine_epsilon(dtype: np.dtype | type[np.floating]) -> float:
    """Get machine epsilon for a given floating-point dtype.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Machine epsilon for the given dtype
    """

    _check_dtype(dtype)

    return float(np.finfo(dtype).eps)


class ToleranceInfo(TypedDict):
    dtype: np.dtype | type[np.floating]
    machine_epsilon: float
    default_tolerance: float
    strict_tolerance: float
    conservative_tolerance: float
    precision_bits: int
    max_value: float
    min_value: float


def get_tolerance_info(
    dtype: np.dtype | type[np.floating],
) -> ToleranceInfo:
    """Get comprehensive tolerance information for a dtype.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Dictionary with tolerance information
    """

    _check_dtype(dtype)

    return {
        "dtype": dtype,
        "machine_epsilon": get_machine_epsilon(dtype),
        "default_tolerance": get_default_tolerance(dtype),
        "strict_tolerance": get_strict_tolerance(dtype),
        "conservative_tolerance": get_conservative_tolerance(dtype),
        "precision_bits": np.finfo(dtype).precision,
        "max_value": float(np.finfo(dtype).max),
        "min_value": float(np.finfo(dtype).tiny),
    }
