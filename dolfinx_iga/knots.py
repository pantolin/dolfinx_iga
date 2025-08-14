from typing import Optional

import numpy as np
import numpy.typing as npt

from .utils.tolerance import get_strict_tolerance, unique_with_tolerance


class KnotsVector:
    """Represents a vector of knots for spline interpolation.

    This class handles the storage and manipulation of knot vectors,
    including functionality for snapping knots to a specified tolerance.


    """

    _knots_w_rep: npt.NDArray[np.floating]
    _unique_knots: npt.NDArray[np.floating]
    _multiplicities: npt.NDArray[np.int_]
    _tol: float

    def __init__(
        self,
        knots_w_rep: npt.NDArray[np.floating],
        snap_knots: bool = True,
        tolerance: Optional[float] = None,
    ) -> None:
        self._knots_w_rep = knots_w_rep

        if self._knots_w_rep.ndim != 1 or self._knots_w_rep.size < 2:
            raise ValueError(
                "Knots vector must be a 1D numpy array with at least two elements."
            )

        if np.any(np.diff(self._knots_w_rep) < 0):
            raise ValueError("Knots must be monotonically increasing.")

        if tolerance is None:
            self._tol = get_strict_tolerance(self.dtype)
        else:
            self._tol = self.dtype.type(tolerance)

        if self._tol <= 0:
            raise ValueError("Tolerance must be a positive number.")

        if snap_knots:
            self._snap_knots()

        self._unique_knots, self._multiplicities = unique_with_tolerance(
            self._knots_w_rep
        )

        if self._unique_knots.size < 2:
            raise ValueError("Knot vector must contain at least one non zero span.")

    def __len__(self) -> int:
        """Get the length of the knots vector."""
        return len(self._knots_w_rep)

    def __getitem__(self, index) -> np.floating:
        """Get the knot value at the specified index."""
        return self._knots_w_rep[index]

    def __repr__(self) -> str:
        """Concise representation of the knots vector."""
        return f"KnotsVector({self._knots_w_rep!r})"

    def __str__(self) -> str:
        """String representation showing knots with repetitions, unique knots, and multiplicities."""
        knots_str = np.array2string(self._knots_w_rep, precision=4, suppress_small=True)
        unique_str = np.array2string(
            self._unique_knots, precision=4, suppress_small=True
        )
        repetitions_str = np.array2string(
            self._multiplicities, precision=4, suppress_small=True
        )
        return f"Knots with repetitions: {knots_str}\nUnique knots: {unique_str}\nMultiplicities: {repetitions_str}"

    @property
    def knots_with_repetitions(self) -> npt.NDArray[np.floating]:
        """Get the knot vector with repetitions."""
        return self._knots_w_rep

    @property
    def unique_knots(self) -> npt.NDArray[np.floating]:
        """Get the unique knot values."""
        return self._unique_knots

    @property
    def multiplicities(self) -> npt.NDArray[np.int_]:
        """Get the multiplicities of the unique knot values."""
        return self._multiplicities

    @property
    def tolerance(self) -> float:
        """Get the tolerance value for floating point comparisons."""
        return self._tol

    @property
    def dtype(self) -> np.dtype:
        """Get the floating point data type of the knot vector."""
        return self._knots_w_rep.dtype

    def _snap_knots(self) -> None:
        result = self._knots_w_rep.copy()
        unique_vals, _ = unique_with_tolerance(
            self._knots_w_rep, custom_tolerance=self._tol
        )

        for unique_val in unique_vals:
            # Find all knots that are within tolerance of this unique value
            mask = np.abs(self._knots_w_rep - unique_val) <= self._tol
            # Set all matching knots to the exact unique value
            result[mask] = unique_val

        self._knots_w_rep = result

    def num_nonzero_spans(self) -> int:
        """Get the number of non-zero spans in the knot vector."""
        return self._unique_knots.size - 1

    def is_open(self, degree: int) -> bool:
        """Check if the knot vector is open at a given degree."""
        if degree < 0:
            raise ValueError("Degree must be non-negative.")
        # An open knot vector has the first and last knots with maximum multiplicity
        return (
            self._multiplicities[0] == degree + 1
            and self._multiplicities[-1] == degree + 1
        )

    def is_uniform(self) -> bool:
        """Check if the knot vector is uniform."""
        # A uniform knot vector has equal spacing between unique knots
        spacing = np.diff(self._unique_knots)
        return bool(np.all(np.isclose(spacing, spacing[0])))

    def find_span(self, values: npt.NDArray[np.floating]) -> npt.NDArray[np.int_]:
        """Find the indices of non-zero spans to which values belong.

        A value v belongs to a non-zero span [u_i, u_{i+1}) if u_i <= v < u_{i+1}.
        Special case: if v equals the last knot of the vector, it belongs to the last span.

        Args:
            values: Array of values to find span indices for

        Returns:
            Array of span indices for each input value

        Raises:
            ValueError: If any value is outside the knot vector domain (within tolerance)
        """
        values = np.asarray(values, dtype=self.dtype)
        tol = self.tolerance

        # Check bounds with tolerance
        min_knot = self._unique_knots[0]
        max_knot = self._unique_knots[-1]

        if np.any(values < (min_knot - tol)) or np.any(values > (max_knot + tol)):
            raise ValueError(
                f"All values must be within the knot vector domain [{min_knot}, {max_knot}] (within tolerance {tol})"
            )

        # Clamp values to valid range
        values = np.clip(values, min_knot, max_knot)

        # Find span indices using searchsorted
        # searchsorted finds insertion points, we need to adjust for our span definition
        span_indices = np.searchsorted(self._unique_knots[1:], values, side="right")

        # Ensure we have an array for consistent indexing
        span_indices = np.atleast_1d(span_indices)

        # Handle the special case where value equals the last knot
        last_knot_mask = np.isclose(values, max_knot, atol=tol)
        last_knot_mask = np.atleast_1d(last_knot_mask)
        span_indices[last_knot_mask] = self.num_nonzero_spans() - 1

        return span_indices.astype(np.int_)


def create_open_uniform_knot_vector(
    degree: int,
    num_intervals: int,
    start: float = 0.0,
    end: float = 1.0,
    continuity: Optional[int] = None,
    dtype: npt.DTypeLike = np.float64,
) -> KnotsVector:
    """Create an open uniform knot vector with specified continuity.

    Args:
        degree: Polynomial degree
        num_intervals: Number of non-zero intervals
        start: Start point of the knot vector. Defaults to 0
        end: End point of the knot vector. Defaults to 1
        continuity: Desired continuity (C^k). If not defined, uses maximum continuity (degree-1)
        dtype: Data type for the knot vector. If None, defaults to np.float64

    Returns:
        KnotsVector: Open uniform knot vector with specified properties

    Raises:
        ValueError: If parameters are invalid
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative.")
    if start >= end:
        raise ValueError("Start must be less than end.")
    if num_intervals < 1:
        raise ValueError("Number of intervals must be at least 1.")

    if continuity is None:
        continuity = degree - 1

    if continuity < -1 or continuity >= degree:
        raise ValueError(
            f"Continuity must be between -1 and {degree - 1} for degree {degree}."
        )

    interior_multiplicity = degree - continuity

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # Build knot vector with repetitions
    knots_list = []

    # First knot with multiplicity (degree + 1) for open condition
    knots_list.extend([start] * degree)

    # Interior knots with specified multiplicity
    for knot in unique_knots:
        knots_list.extend([knot] * interior_multiplicity)

    # Last knot with multiplicity (degree + 1) for open condition
    knots_list.extend([end] * degree)

    knots_w_rep = np.array(knots_list, dtype=dtype)

    return KnotsVector(knots_w_rep, snap_knots=False)
