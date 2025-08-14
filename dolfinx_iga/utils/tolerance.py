"""Tolerance utilities for floating-point comparisons in IGA applications."""

from typing import Optional

import numpy as np


def get_default_tolerance(dtype: np.dtype) -> float:
    """Get a reasonable default tolerance for floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Recommended tolerance value for the given dtype

    Examples:
        >>> get_default_tolerance(np.float32)
        1e-06
        >>> get_default_tolerance(np.float64)
        1e-12
    """
    if dtype == np.float16:
        return 1e-3
    elif dtype == np.float32:
        return 1e-6
    elif dtype == np.float64:
        return 1e-12
    elif dtype == np.longdouble:
        return 1e-15
    else:
        # Fallback for other floating types
        return float(np.finfo(dtype).eps * 1000)


def get_strict_tolerance(dtype: np.dtype) -> float:
    """Get a strict tolerance for high-precision floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Strict tolerance value for the given dtype (typically for parametric coordinates)
    """
    if dtype == np.float16:
        return 1e-4
    elif dtype == np.float32:
        return 1e-7
    elif dtype == np.float64:
        return 1e-15
    elif dtype == np.longdouble:
        return 1e-18
    else:
        # Fallback for other floating types
        return float(np.finfo(dtype).eps * 100)


def get_conservative_tolerance(dtype: np.dtype) -> float:
    """Get a conservative tolerance for robust floating-point comparisons.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Conservative tolerance value for the given dtype
    """
    if dtype == np.float16:
        return 1e-2
    elif dtype == np.float32:
        return 1e-5
    elif dtype == np.float64:
        return 1e-10
    elif dtype == np.longdouble:
        return 1e-12
    else:
        # Fallback for other floating types
        return float(np.finfo(dtype).eps * 10000)


def are_close(
    a: float | np.floating,
    b: float | np.floating,
    dtype: np.dtype,
    tolerance_type: str = "default",
) -> bool:
    """Check if two floating-point values are close within tolerance.

    Args:
        a: First value
        b: Second value
        dtype: NumPy floating-point data type to determine tolerance
        tolerance_type: Type of tolerance ("default", "strict", "conservative")

    Returns:
        True if values are within tolerance, False otherwise

    Examples:
        >>> are_close(1.0, 1.000001, np.float32)
        True
        >>> are_close(1.0, 1.01, np.float32)
        False
    """
    if tolerance_type == "strict":
        tol = get_strict_tolerance(dtype)
    elif tolerance_type == "conservative":
        tol = get_conservative_tolerance(dtype)
    else:  # default
        tol = get_default_tolerance(dtype)

    return abs(float(a) - float(b)) <= tol


def are_arrays_close(
    arr1: np.ndarray, arr2: np.ndarray, tolerance_type: str = "default"
) -> bool:
    """Check if two arrays are close within tolerance based on their dtype.

    Args:
        arr1: First array
        arr2: Second array
        tolerance_type: Type of tolerance ("default", "strict", "conservative")

    Returns:
        True if all corresponding elements are within tolerance

    Raises:
        ValueError: If arrays have different shapes or dtypes
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")

    if arr1.dtype != arr2.dtype:
        raise ValueError("Arrays must have the same dtype")

    if tolerance_type == "strict":
        tol = get_strict_tolerance(arr1.dtype)
    elif tolerance_type == "conservative":
        tol = get_conservative_tolerance(arr1.dtype)
    else:  # default
        tol = get_default_tolerance(arr1.dtype)

    return np.allclose(arr1, arr2, atol=tol, rtol=0)


def unique_with_tolerance(
    arr: np.ndarray,
    tolerance_type: str = "default",
    custom_tolerance: Optional[float] = None,
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


def get_machine_epsilon(dtype: np.dtype) -> float:
    """Get machine epsilon for a given floating-point dtype.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Machine epsilon for the given dtype
    """
    return float(np.finfo(dtype).eps)


def get_tolerance_info(dtype: np.dtype) -> dict:
    """Get comprehensive tolerance information for a dtype.

    Args:
        dtype: NumPy floating-point data type

    Returns:
        Dictionary with tolerance information
    """
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
