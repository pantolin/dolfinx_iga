"""Tolerance utilities for floating-point comparisons in IGA applications."""

from typing import Optional

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
    values: list[np.float16 | np.float32 | np.float64 | np.longdouble],
) -> np.floating:
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


def get_default_tolerance(dtype: np.dtype | type[np.floating]) -> np.floating:
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
        [np.float16(1e-3), np.float32(1e-6), np.float64(1e-12), np.longdouble(1e-15)],
    )


def get_strict_tolerance(dtype: np.dtype | type[np.floating]) -> np.floating:
    """Get a strict tolerance for high-precision floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Strict tolerance value for the given dtype (typically for parametric coordinates)
    """

    return _get_tolerance(
        dtype,
        [np.float16(1e-4), np.float32(1e-7), np.float64(1e-15), np.longdouble(1e-18)],
    )


def get_conservative_tolerance(dtype: np.dtype | type[np.floating]) -> np.floating:
    """Get a conservative tolerance for robust floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Conservative tolerance value for the given dtype
    """

    return _get_tolerance(
        dtype,
        [np.float16(1e-2), np.float32(1e-5), np.float64(1e-10), np.longdouble(1e-12)],
    )


def unique_with_tolerance(
    arr: np.ndarray,
    tolerance_type: str = "default",
    custom_tolerance: Optional[np.floating] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find unique values in array within tolerance.

    Args:
        arr: Input array
        tolerance_type: Type of tolerance ("default", "strict", "conservative")
        custom_tolerance: Custom tolerance value (overrides tolerance_type)

    Returns:
        Tuple of (unique_values, counts)

    Examples:
        >>> arr = np.array([1.0, 1.000001, 2.0, 2.000001], dtype=np.float32)
        >>> unique_vals, counts = unique_with_tolerance(arr)
        >>> len(unique_vals)
        2
    """
    if custom_tolerance is not None:
        tol = custom_tolerance
    elif tolerance_type == "strict":
        tol = get_strict_tolerance(arr.dtype)
    elif tolerance_type == "conservative":
        tol = get_conservative_tolerance(arr.dtype)
    else:  # default
        tol = get_default_tolerance(arr.dtype)

    # Round to tolerance precision for grouping
    scale = 1.0 / tol
    rounded_arr = np.round(arr * scale) / scale
    unique, counts = np.unique(rounded_arr, return_counts=True)

    return unique, counts


def get_machine_epsilon(dtype: np.dtype | type[np.floating]) -> float:
    """Get machine epsilon for a given floating-point dtype.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Machine epsilon for the given dtype
    """

    _check_dtype(dtype)

    return float(np.finfo(dtype).eps)


def get_tolerance_info(dtype: np.dtype | type[np.floating]) -> dict:
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
