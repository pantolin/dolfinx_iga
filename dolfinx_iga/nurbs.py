"""
NURBS (Non-Uniform Rational B-Spline) curve and surface implementations.

This module provides NURBS functionality for isogeometric analysis,
extending B-splines with rational basis functions for exact representation
of conic sections and other curved geometries.
"""

from typing import Optional, Union

import numpy as np

from .bspline import BSplineCurve, BSplineSurface


class NURBSCurve:
    """
    NURBS curve implementation (rational B-spline curve).

    Parameters
    ----------
    control_points : np.ndarray
        Control points array of shape (n_points, dimension)
    weights : np.ndarray
        Weights for rational basis functions
    degree : int
        Degree of the NURBS curve
    knot_vector : np.ndarray, optional
        Knot vector. If None, uniform knot vector is generated
    """

    def __init__(
        self,
        control_points: np.ndarray,
        weights: np.ndarray,
        degree: int,
        knot_vector: Optional[np.ndarray] = None,
    ):
        self.control_points = np.asarray(control_points)
        self.weights = np.asarray(weights)
        self.degree = degree

        # Create homogeneous control points (weighted)
        weighted_points = self.control_points * self.weights[:, np.newaxis]
        homogeneous_points = np.column_stack([weighted_points, self.weights])

        # Use B-spline curve for computation in homogeneous space
        self._bspline = BSplineCurve(homogeneous_points, degree, knot_vector)

    @property
    def knot_vector(self) -> np.ndarray:
        """Get the knot vector."""
        return self._bspline.knot_vector

    def evaluate(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the NURBS curve at parameter value(s) u.

        Parameters
        ----------
        u : float or np.ndarray
            Parameter value(s) where to evaluate the curve

        Returns
        -------
        np.ndarray
            Curve point(s) at parameter value(s) u
        """
        # Evaluate in homogeneous space
        homogeneous_points = self._bspline.evaluate(u)

        if homogeneous_points.ndim == 1:
            # Single point
            w = homogeneous_points[-1]
            return homogeneous_points[:-1] / w
        else:
            # Multiple points
            w = homogeneous_points[:, -1]
            return homogeneous_points[:, :-1] / w[:, np.newaxis]

    def derivative(self, u: Union[float, np.ndarray], order: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of the NURBS curve using quotient rule.

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
        # For NURBS derivatives, we need to apply the quotient rule
        # This is a simplified implementation for first-order derivatives
        if order != 1:
            raise NotImplementedError("Higher order derivatives not yet implemented")

        # Get homogeneous derivatives
        homogeneous_derivs = self._bspline.derivative(u, order)
        homogeneous_points = self._bspline.evaluate(u)

        if homogeneous_points.ndim == 1:
            # Single point
            w = homogeneous_points[-1]
            w_deriv = homogeneous_derivs[-1]
            numerator_deriv = homogeneous_derivs[:-1]
            numerator = homogeneous_points[:-1]

            return (numerator_deriv * w - numerator * w_deriv) / (w * w)
        else:
            # Multiple points
            w = homogeneous_points[:, -1]
            w_deriv = homogeneous_derivs[:, -1]
            numerator_deriv = homogeneous_derivs[:, :-1]
            numerator = homogeneous_points[:, :-1]

            return (
                numerator_deriv * w[:, np.newaxis] - numerator * w_deriv[:, np.newaxis]
            ) / (w[:, np.newaxis] ** 2)


class NURBSSurface:
    """
    NURBS surface implementation (rational B-spline surface).

    Parameters
    ----------
    control_points : np.ndarray
        Control points array of shape (n_u, n_v, dimension)
    weights : np.ndarray
        Weights for rational basis functions of shape (n_u, n_v)
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
        weights: np.ndarray,
        degree_u: int,
        degree_v: int,
        knot_vector_u: Optional[np.ndarray] = None,
        knot_vector_v: Optional[np.ndarray] = None,
    ):
        self.control_points = np.asarray(control_points)
        self.weights = np.asarray(weights)
        self.degree_u = degree_u
        self.degree_v = degree_v

        # Create homogeneous control points (weighted)
        weighted_points = self.control_points * self.weights[:, :, np.newaxis]
        homogeneous_points = np.concatenate(
            [weighted_points, self.weights[:, :, np.newaxis]], axis=2
        )

        # Use B-spline surface for computation in homogeneous space
        self._bspline = BSplineSurface(
            homogeneous_points, degree_u, degree_v, knot_vector_u, knot_vector_v
        )

    @property
    def knot_vector_u(self) -> np.ndarray:
        """Get the knot vector in u direction."""
        return self._bspline.knot_vector_u

    @property
    def knot_vector_v(self) -> np.ndarray:
        """Get the knot vector in v direction."""
        return self._bspline.knot_vector_v

    def evaluate(
        self, u: Union[float, np.ndarray], v: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Evaluate the NURBS surface at parameter values (u, v).

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
        # Evaluate in homogeneous space
        homogeneous_points = self._bspline.evaluate(u, v)

        # Project back to Cartesian space
        w = homogeneous_points[..., -1]
        cartesian_points = homogeneous_points[..., :-1] / w[..., np.newaxis]

        return cartesian_points
