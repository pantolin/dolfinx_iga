"""Knot vector generation utilities for B-splines.

This module provides functions to create various types of knot vectors including
uniform open, uniform periodic, and cardinal B-spline knot vectors with
configurable continuity and domain parameters.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from ..utils.types import Float_32_64


def _validate_knot_input(
    num_intervals: np.int_,
    degree: np.int_,
    continuity: np.int_,
    start: np.floating,
    end: np.floating,
    dtype: np.dtype,
) -> None:
    """Validate input parameters for knot vector generation.

    Args:
        num_intervals (np.int_): Number of intervals in the domain.
        degree (np.int_): B-spline degree.
        continuity (np.int_): Continuity level at interior knots.
        start (np.floating): Start value of the domain.
        end (np.floating): End value of the domain.
        dtype (np.dtype): Data type for the knot vector.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if start >= end:
        raise ValueError("start must be less than end")

    if num_intervals < 0:
        raise ValueError("num_intervals must be non-negative")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    if continuity < -1 or continuity >= degree:
        raise ValueError(
            f"Continuity must be between -1 and {degree - 1} for degree {degree}."
        )

    if dtype not in (
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.float32,
        np.float64,
    ):
        raise ValueError("dtype must be float64 or float32")


def _get_ends_and_type(
    start: Optional[Float_32_64 | float] = None,
    end: Optional[Float_32_64 | float] = None,
    dtype: Optional[np.dtype] = None,
) -> tuple[np.floating, np.floating, np.dtype]:
    """Get the start, end, and dtype for a knot vector.

    Args:
        start (Optional[Float_32_64 | float]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[Float_32_64 | float]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        tuple[np.floating, np.floating, np.dtype]: Tuple of (start, end, dtype).

    Raises:
        ValueError: If start and end have different dtypes, or if end <= start.
    """

    if dtype is None:
        if start is None and end is None:
            dtype = np.float64
        elif start is not None:
            start = np.array(start).item()
            if isinstance(start, float):
                start = np.float64(start)
            dtype = start.dtype
            if end is not None:
                end = np.array(end).item()
                if isinstance(end, float):
                    end = np.float64(end)
                dtype = end.dtype
                if end.dtype != dtype:
                    raise ValueError("start and end must have the same dtype")
        else:  # if end is not None:
            end = np.array(end).item()
            dtype = end.dtype

    else:
        # When dtype is provided, validate that inputs match the expected dtype
        if start is not None:
            start_array = np.array(start)
            if start_array.dtype != dtype:
                raise ValueError(f"start must be of type dtype {dtype}")
            start = start_array.item()
        if end is not None:
            end_array = np.array(end)
            if end_array.dtype != dtype:
                raise ValueError(f"end must be of type dtype {dtype}")
            end = end_array.item()

        # Check that start and end have the same dtype if both are provided
        if start is not None and end is not None:
            start_array = np.array(start)
            end_array = np.array(end)
            if start_array.dtype != end_array.dtype:
                raise ValueError("start and end must have the same dtype")

    if start is None:
        start = dtype(0.0)

    if end is None:
        end = dtype(1.0)

    if end <= start:
        raise ValueError("end must be greater than start")

    return start, end, dtype


def create_uniform_open_knot_vector(
    num_intervals: np.int_,
    degree: np.int_,
    continuity: Optional[np.int_] = None,
    start: Optional[Float_32_64 | float] = None,
    end: Optional[Float_32_64 | float] = None,
    dtype: Optional[np.dtype] = None,
) -> npt.NDArray[np.floating]:
    """Create a uniform open knot vector.

    An open knot vector has the first and last knots repeated (degree+1) times,
    ensuring the B-spline interpolates the first and last control points.

    Args:
        num_intervals (np.int_): Number of intervals in the domain. Must be non-negative.
        degree (np.int_): B-spline degree. Must be non-negative.
        continuity (Optional[np.int_]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        start (Optional[Float_32_64 | float]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[Float_32_64 | float]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Open knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_open_knot_vector(2, 2, start=0.0, end=1.0)
        array([0., 0., 0., 0.5, 1., 1., 1.])
    """

    start, end, dtype = _get_ends_and_type(start, end, dtype)

    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        start,
        end,
        dtype,
    )

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # First knot with multiplicity (degree + 1) for open condition
    knots = np.array([start] * (degree + 1), dtype)

    # Interior knots with specified multiplicity
    interior_multiplicity = degree - continuity
    for knot in unique_knots[1:-1]:
        knots = np.append(knots, [knot] * interior_multiplicity)

    # Last knot with multiplicity (degree + 1) for open condition
    knots = np.append(knots, [end] * (degree + 1))

    return knots


def create_uniform_periodic_knot_vector(
    num_intervals: np.int_,
    degree: np.int_,
    continuity: Optional[np.int_] = None,
    start: Optional[Float_32_64 | float] = None,
    end: Optional[Float_32_64 | float] = None,
    dtype: Optional[np.dtype] = None,
) -> npt.NDArray[np.floating]:
    """Create a uniform periodic knot vector.

    A periodic knot vector extends beyond the domain boundaries to ensure
    periodicity of the B-spline basis functions.

    Args:
        num_intervals (np.int_): Number of intervals in the domain. Must be non-negative.
        degree (np.int_): B-spline degree. Must be non-negative.
        continuity (Optional[np.int_]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        start (Optional[Float_32_64 | float]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[Float_32_64 | float]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Periodic knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_periodic_knot_vector(2, 2, start=0.0, end=1.0)
        array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """

    start, end, dtype = _get_ends_and_type(start, end, dtype)
    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        start,
        end,
        dtype,
    )

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # Build knot vector with repetitions
    knots = np.array([], dtype=dtype)

    multiplicity = degree - continuity

    # Starting periodic knots.
    length = (end - start) / num_intervals
    knots = np.linspace(
        start - length * (degree - multiplicity + 1),
        start,
        degree + 2 - multiplicity,
        dtype=dtype,
    )[:-1]

    # Interior knots with specified multiplicity
    for knot in unique_knots:
        knots = np.append(knots, [knot] * multiplicity)

    # End periodic knots.
    knots = np.append(
        knots,
        np.linspace(
            end,
            end + length * (degree - multiplicity + 1),
            degree + 2 - multiplicity,
            dtype=dtype,
        )[1:],
    )

    return knots


def create_cardinal_Bspline_knot_vector(
    num_intervals: np.int_,
    degree: np.int_,
    dtype: np.dtype = np.float64,
) -> npt.NDArray[np.floating]:
    """Create a knot vector for cardinal B-spline basis functions.

    Cardinal B-splines are B-splines defined on uniform knot vectors with
    maximum continuity, where the basis functions in the central region
    have the same shape and are translated versions of each other.

    Args:
        num_intervals (np.int_): Number of intervals in the domain. Must be at least 1.
        degree (np.int_): B-spline degree. Must be non-negative.
        dtype (np.dtype): Data type for the knot vector. Defaults to np.float64.

    Returns:
        npt.NDArray[np.floating]: Cardinal B-spline knot vector with uniform spacing.

    Raises:
        ValueError: If num_intervals < 1, degree < 0, or dtype is not float32/float64.

    Example:
        >>> create_cardinal_Bspline_knot_vector(2, 2)
        array([-2., -1.,  0.,  1.,  2.,  3., 4.])
    """

    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    start = 0.0
    end = float(num_intervals)
    if dtype is not None:
        start = dtype(start)
        end = dtype(end)

    return create_uniform_periodic_knot_vector(
        num_intervals, degree, continuity=degree - 1, start=start, end=end, dtype=dtype
    )
