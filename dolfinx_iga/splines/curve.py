"""B-spline curve representation and evaluation.

This module provides a BsplineCurve class for representing and evaluating
B-spline curves in arbitrary dimensions, supporting both polynomial and
rational (NURBS) curves.
"""

import numpy as np
from numpy.typing import NDArray

from ..utils.types import FloatLike_32_64
from .basis_1D import evaluate_Bspline_basis
from .bspline_1D import Bspline1D


class BsplineCurve:
    """A class representing a B-spline curve in arbitrary dimensions.

    This class supports both polynomial B-spline curves and rational B-spline
    curves (NURBS). The curve is defined by a B-spline space and control points.

    Attributes:
        _space (Bspline1D): B-spline space defining the basis functions.
        _control_points (NDArray[np.floating]): Control points defining the curve.
        _rational (bool): Whether the curve is rational (NURBS).
    """

    def __init__(
        self,
        space: Bspline1D,
        control_points: NDArray[np.floating],
        is_rational: bool = False,
    ):
        """Initialize a B-spline curve.

        Args:
            space (Bspline1D): B-spline space defining the basis functions.
            control_points (NDArray[np.floating]): Control points as a 2D array
                with shape (num_control_points, geom_dimension + weight) where the
                last column contains weights for rational curves.
            is_rational (bool): Whether the curve is rational (NURBS).
                Defaults to False.

        Raises:
            ValueError: If control points are invalid or don't match the space.
        """
        self._space = space

        if np.issubdtype(control_points.dtype, np.integer):
            self._control_points = control_points.astype(np.float64)
        else:
            self._control_points = control_points

        self._rational = is_rational

        self._validate()

    @property
    def space(self) -> Bspline1D:
        """Get the B-spline space.

        Returns:
            Bspline1D: The B-spline space defining the basis functions.
        """
        return self._space

    @property
    def control_points(self) -> NDArray[np.floating]:
        """Get the control points.

        Returns:
            NDArray[np.floating]: The control points as a 2D array.
        """
        return self._control_points

    @property
    def rational(self) -> bool:
        """Check if the curve is rational (NURBS).

        Returns:
            bool: True if the curve is rational, False otherwise.
        """
        return self._rational

    @property
    def geom_dim(self) -> np.int_:
        """Get the geometric dimension of the curve.

        Returns:
            np.int_: The geometric dimension (excluding weights for rational curves).
        """
        return self.control_points.shape[1] - (1 if self._rational else 0)

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the control points.

        Returns:
            np.dtype: The numpy data type of the control points.
        """
        return self.control_points.dtype

    @property
    def num_control_points(self) -> np.int_:
        """Get the number of control points.

        Returns:
            np.int_: The number of control points.
        """
        return self.control_points.shape[0]

    @property
    def periodic(self) -> bool:
        """Check if the curve is periodic.

        Returns:
            bool: True if the curve is periodic, False otherwise.
        """
        return self.space.periodic

    @property
    def degree(self) -> np.int_:
        """Get the degree of the B-spline curve.

        Returns:
            np.int_: The degree of the B-spline.
        """
        return self.space.degree

    def get_domain(self) -> tuple[np.floating, np.floating]:
        """Get the domain of the B-spline curve.

        Returns:
            tuple[np.floating, np.floating]: Tuple of (start, end) defining the domain.
        """
        return self._space.get_domain()

    def _validate(self) -> None:
        """Validate the curve parameters.

        Raises:
            ValueError: If control points are invalid or don't match the space.
        """
        if self.control_points.ndim != 2:
            raise ValueError("Control points must be a 2D array")
        if self.control_points.shape[0] != self.space.get_num_basis():
            raise ValueError(
                "Number of control points must match the number of basis functions"
            )

        if self.control_points.shape[1] == 0:
            raise ValueError("Invalid number of coordinates")
        elif self.rational and self.control_points.shape[1] < 2:
            raise ValueError("Invalid number of coordinates for rational curves")

        if self.dtype != self.space.dtype:
            raise ValueError("Control points and space must have the same dtype")

    def evaluate(self, pts: FloatLike_32_64) -> NDArray[np.floating]:
        """Evaluate the B-spline curve at a given parameter value.

        Args:
            k (np.floating): Parameter value at which to evaluate the curve.
                Must be within the curve's domain.

        Returns:
            NDArray[np.floating]: Point on the curve as a 1D array of length
                equal to the curve's geometric dimension.

        Raises:
            ValueError: If the parameter is outside the curve's domain.

        Example:
            >>> from dolfinx_iga.splines.knots import create_uniform_open_knot_vector
            >>> knots = create_uniform_open_knot_vector(2, 2)
            >>> space = Bspline1D(knots, 2)
            >>> control_points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
            >>> curve = BsplineCurve(space, control_points)
            >>> curve.evaluate([0.5, 0.75, 0.85])
            array([[1.5   0.5  ],
                   [2.125 0.375],
                   [2.445 0.535]])
        """
        basis, first_basis_ids = evaluate_Bspline_basis(self.space, pts)
        # Create a 2D array: each row is [id, id+1, ..., id+degree]
        bases_ids = first_basis_ids[:, None] + np.arange(self.degree + 1)

        # for the periodic case
        bases_ids = np.mod(bases_ids, self.num_control_points)
        ctrl_pts = self.control_points[bases_ids]

        pts_eval = np.einsum("ij,ijk->ik", basis, ctrl_pts)

        # This should be a vectorized version of:
        # # For each row, multiply basis[i] (shape: (degree+1,)) with control_points[bases_ids[i]] (shape: (degree+1, geom_dim))
        # pts_eval = np.empty((n_pts, self.control_points.shape[1]), dtype=self.dtype)
        # for i in range(n_pts):
        #     pts_eval[i] = basis[i] @ self.control_points[bases_ids[i]]

        if self.rational:
            # For rational curves, divide by the last coordinate (weight)
            weights = pts_eval[:, -1:]  # Keep last column as weights
            coords = pts_eval[:, :-1]  # All but last column as coordinates
            result = coords / weights
        else:
            result = pts_eval

        # If single point evaluation, return 1D array
        if result.shape[0] == 1:
            return result[0]
        return result
