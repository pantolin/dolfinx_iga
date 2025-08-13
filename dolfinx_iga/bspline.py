"""
B-spline curve and surface implementations.

This module provides B-spline functionality for isogeometric analysis,
including curve and surface evaluation, derivatives, and basis functions.
"""

from typing import Optional

import numpy as np

from .utils.basis_functions import bspline_basis, bspline_basis_derivatives
from .utils.knot_vector_utils import generate_uniform_knot_vector, validate_knot_vector


class BSplineCurve:
    """
    B-spline curve implementation.

    Parameters
    ----------
    control_points : np.ndarray
        Control points array of shape (n_points, dimension)
    degree : int
        Degree of the B-spline curve
    knot_vector : np.ndarray, optional
        Knot vector. If None, uniform knot vector is generated
    """

    def __init__(
        self,
        control_points: np.ndarray,
        degree: int,
        knot_vector: Optional[np.ndarray] = None,
    ):
        self.control_points = np.asarray(control_points)
        self.degree = degree
        self.n_control_points = len(self.control_points)

        if knot_vector is None:
            self.knot_vector = generate_uniform_knot_vector(
                self.n_control_points, degree
            )
        else:
            self.knot_vector = np.asarray(knot_vector)
            validate_knot_vector(self.knot_vector, self.n_control_points, degree)

    def evaluate(self, u: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the B-spline curve at parameter value(s) u.

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s) where to evaluate the curve

        Returns
        -------
        np.ndarray
            Curve point(s) at parameter value(s) u
        """
        u_array = np.atleast_1d(u)
        points = np.zeros((len(u_array), self.control_points.shape[1]))

        for i, u_val in enumerate(u_array):
            basis_vals = bspline_basis(
                u_val, self.degree, self.knot_vector, self.n_control_points
            )
            points[i] = np.sum(basis_vals[:, np.newaxis] * self.control_points, axis=0)

        return points.squeeze() if len(u_array) == 1 else points

    def derivative(self, u: float | np.ndarray, order: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of the B-spline curve.

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s) where to evaluate derivatives
        order : int
            Order of derivative (default: 1)

        Returns
        -------
        np.ndarray
            Derivative values at parameter value(s) u
        """
        u_array = np.atleast_1d(u)
        derivatives = np.zeros((len(u_array), self.control_points.shape[1]))

        for i, u_val in enumerate(u_array):
            basis_derivs = bspline_basis_derivatives(
                u_val, self.degree, self.knot_vector, self.n_control_points, order
            )
            derivatives[i] = np.sum(
                basis_derivs[order, :, np.newaxis] * self.control_points, axis=0
            )

        return derivatives.squeeze() if len(u_array) == 1 else derivatives


class BSplineSurface:
    """
    B-spline surface implementation.

    Parameters
    ----------
    control_points : np.ndarray
        Control points array of shape (n_u, n_v, dimension)
    degree_u : int
        Degree in u direction
    degree_v : int
        Degree in v direction
    knot_vector_u : np.ndarray, optional
        Knot vector in u direction
    knot_vector_v : np.ndarray, optional
        Knot vector in v direction
    """

    def __init__(
        self,
        control_points: np.ndarray,
        degree_u: int,
        degree_v: int,
        knot_vector_u: Optional[np.ndarray] = None,
        knot_vector_v: Optional[np.ndarray] = None,
    ):
        self.control_points = np.asarray(control_points)
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.n_u, self.n_v = self.control_points.shape[:2]

        if knot_vector_u is None:
            self.knot_vector_u = generate_uniform_knot_vector(self.n_u, degree_u)
        else:
            self.knot_vector_u = np.asarray(knot_vector_u)
            validate_knot_vector(self.knot_vector_u, self.n_u, degree_u)

        if knot_vector_v is None:
            self.knot_vector_v = generate_uniform_knot_vector(self.n_v, degree_v)
        else:
            self.knot_vector_v = np.asarray(knot_vector_v)
            validate_knot_vector(self.knot_vector_v, self.n_v, degree_v)

    def evaluate(self, u: float | np.ndarray, v: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the B-spline surface at parameter values (u, v).

        Parameters
        ----------
        u : float or np.ndarray
            Parameter values in u direction
        v : float or np.ndarray
            Parameter values in v direction

        Returns
        -------
        np.ndarray
            Surface points at parameter values (u, v)
        """
        u_array = np.atleast_1d(u)
        v_array = np.atleast_1d(v)

        # Create meshgrid for evaluation
        U, V = np.meshgrid(u_array, v_array, indexing="ij")
        points = np.zeros((*U.shape, self.control_points.shape[2]))

        for i in range(len(u_array)):
            for j in range(len(v_array)):
                basis_u = bspline_basis(
                    U[i, j], self.degree_u, self.knot_vector_u, self.n_u
                )
                basis_v = bspline_basis(
                    V[i, j], self.degree_v, self.knot_vector_v, self.n_v
                )

                # Tensor product of basis functions
                basis_uv = np.outer(basis_u, basis_v)

                # Evaluate surface point
                points[i, j] = np.sum(
                    basis_uv[:, :, np.newaxis] * self.control_points, axis=(0, 1)
                )

        return points.squeeze()
