"""Basis function evaluation for various polynomial bases in 1D.

This module provides functions to evaluate different types of basis functions
including Bernstein, cardinal B-spline, Lagrange, and monomial bases, as well
as B-spline basis evaluation.
"""

from math import factorial
from typing import TYPE_CHECKING

import numpy as np
from basix import CellType, ElementFamily, LagrangeVariant, create_element
from numpy import typing as npt
from scipy.special import binom

from ..utils.types import FloatArray_32_64, FloatLike_32_64, IntArray
from .bspline_1D_impl import evaluate_basis_Cox_de_Boor_impl, is_in_domain_impl

if TYPE_CHECKING:
    from .bspline_1D import Bspline1D


def _prepare_pts_for_evaluation(pts: FloatLike_32_64) -> FloatArray_32_64:
    """Prepare points for basis function evaluation.

    Ensures points are a 1D numpy array with proper shape for evaluation.

    Args:
        pts (FloatLike_32_64): Input points (scalar, array, or list).

    Returns:
        FloatArray_32_64: 1D numpy array of points ready for evaluation.
    """
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if np.issubdtype(pts.dtype, np.integer):
        pts = pts.astype(np.float64)

    if pts.ndim == 0:
        pts = np.array([pts], dtype=pts.dtype)
    elif pts.ndim > 1:
        raise ValueError("pts must be a 1D array")

    return pts


def evaluate_Bernstein_basis(degree: np.int_, pts: FloatLike_32_64) -> FloatArray_32_64:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        degree (np.int_): Degree of the Bernstein polynomials. Must be non-negative.
        pts (FloatLike_32_64): Evaluation points.

    Returns:
        FloatArray_32_64: Array of shape (number points, degree+1)
            containing Bernstein basis function values at each point.

    Raises:
        ValueError: If degree is negative.

    Example:
        >>> evaluate_Bernstein_basis(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.    , 0.    , 0.    ],
               [0.25  , 0.5   , 0.25  ],
               [0.0625, 0.375 , 0.5625],
               [0.    , 0.    , 1.    ]])
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")

    pts = _prepare_pts_for_evaluation(pts)

    dtype = pts.dtype

    i = np.arange(degree + 1, dtype=dtype)

    coeffs = np.array([binom(degree, j) for j in range(degree + 1)], dtype=dtype)

    powers_0 = np.power(pts[..., np.newaxis], i)  # shape (..., degree+1)
    powers_1 = np.power(dtype.type(1.0) - pts[..., np.newaxis], dtype.type(degree) - i)
    return coeffs * powers_0 * powers_1  # broadcasted: (..., degree+1)


def evaluate_cardinal_Bspline_basis(
    degree: int,
    pts: FloatLike_32_64,
) -> FloatArray_32_64:
    r"""
    Evaluate the cardinal B-spline basis polynomials of given degree at given points.

    The cardinal B-spline basis is the set of B-spline basis functions defined
    on an interval of maximum continuity that has degree-1 contiguous
    knot spans on each side with the same length as the interval itself.
    These basis functions appear in the central knot spans
    in the case of maximum regularity uniform knot vectors.

    Explicit expression:
    \[
    B_{p,i}(t) = (1/p!) * sum_{j=0}^{p-i} binom(p+1, j) * (-1)^j * (t + p - i - j)^p
    \]
    where \( B_{p,i}(t) \) is the B-spline basis function of degree \( p \) and index \( i \) at point \( t \), and \( binom(a, b) \) is the binomial coefficient.

    Args:
        degree (int): Degree of the B-spline basis. Must be non-negative.
        pts (FloatLike_32_64): Evaluation points.

    Returns:
        FloatArray_32_64:
            Array of shape (number pts, degree+1) with \( B_{p,i}(t) \)
            for i=0,...,degree at each point \( t \).

    Raises:
        ValueError: If provided degree is negative.

    Example:
        >>> evaluate_cardinal_Bspline_basis(2, [0.0, 0.5, 1.0])
        array([[0.5    , 0.5    , 0.     ],
               [0.125  , 0.75   , 0.125  ],
               [0.03125, 0.6875 , 0.28125],
               [0.     , 0.5    , 0.5    ]])
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")

    pts = _prepare_pts_for_evaluation(pts)

    dtype = pts.dtype

    basis = np.zeros((pts.size, degree + 1), dtype)
    deg = dtype.type(degree)

    fact = factorial(degree)
    for i in range(degree + 1):
        js = np.arange(degree - i + 1)
        coeffs = np.array(
            [binom(degree + 1, j) / fact * (-1) ** j for j in js],
            dtype=dtype,
        )
        basis[:, i] = np.power(pts[..., np.newaxis] + deg - i - js, deg) @ coeffs

    return basis


def evaluate_Lagrange_basis(
    degree: np.int_,
    pts: FloatLike_32_64,
    lagrange_variant: LagrangeVariant = LagrangeVariant.equispaced,
) -> FloatArray_32_64:
    """Evaluate the Lagrange basis polynomials of the given degree and variant at the given points.

    Note:
        The evaluated basis follows the basix ordering of the basis functions.
        See for instance https://defelement.org/elements/examples/interval-lagrange-equispaced-3.html

    Args:
        degree (np.int_): Degree of the Lagrange polynomials.
        pts (FloatLike_32_64): Evaluation points.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, GLL, etc). Defaults to LagrangeVariant.equispaced.

    Returns:
        FloatArray_32_64: Array of shape (number pts, degree+1) containing Lagrange
            basis function values at each point.

    Example:
        >>> evaluate_Lagrange_basis(2, [0.0, 0.5, 0.75, 1.0])
        array([[ 1.00000000e+00, -8.75411412e-18,  3.06126334e-17],
               [-5.11340942e-17,  4.37705706e-18,  1.00000000e+00],
               [-1.25000000e-01,  3.75000000e-01,  7.50000000e-01],
               [-6.42652654e-17,  1.00000000e+00,  3.06126334e-17]])
    """

    if degree < 1:
        raise ValueError("Lagrange basis degree must be at least 1")

    element = create_element(
        ElementFamily.P, CellType.interval, degree, lagrange_variant
    )

    pts = _prepare_pts_for_evaluation(pts).reshape(-1, 1)

    return element.tabulate(0, pts)[0, :, :, 0]


def evaluate_monomial_basis(degree: np.int_, pts: FloatLike_32_64) -> FloatArray_32_64:
    """Evaluate the monomial basis functions up to the given degree at the given points.

    Args:
        degree (np.int_): Maximum degree of monomials. Must be non-negative.
        pts (FloatLike_32_64): Evaluation points.

    Returns:
        FloatArray_32_64: Array of shape (number pts, degree+1) containing monomial
            basis function values [1, t, t^2, ..., t^degree] at each point.

    Raises:
        ValueError: If degree is negative.

    Example:
        >>> evaluate_monomial_basis(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.  , 0.  , 0.  ],
               [1.  , 0.5 , 0.25],
               [1.  , 0.75, 0.5625],
               [1.  , 1.  , 1.  ]])
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    pts = _prepare_pts_for_evaluation(pts)

    dtype = pts.dtype

    i = np.arange(degree + 1, dtype=dtype)

    return np.power(pts[..., np.newaxis], i)


def _evaluate_Bspline_basis_Bernstein_like(
    spline: "Bspline1D",
    pts: FloatArray_32_64,
) -> tuple[FloatArray_32_64, IntArray]:
    """Evaluate B-spline basis functions when they reduce to Bernstein polynomials.

    This function is used when the B-spline has Bézier-like knots, allowing
    direct evaluation using Bernstein basis functions.

    Args:
        spline (Bspline1D): B-spline object with Bézier-like knots.
        pts (FloatArray_32_64): Evaluation points.

    Returns:
        tuple[FloatArray_32_64, IntArray]: Tuple of (basis_values, first_basis_indices)
            where basis_values is an array of shape (number pts, degree+1) that contains
            the Bernstein basis function values and first_basis_indices contains the indices of the first non-zero basis function (all zeros for Bézier-like B-splines).

    Raises:
        AssertionError: If the B-spline does not have Bézier-like knots.
    """
    assert spline.has_Bezier_like_knots()

    # map the points to the reference interval [0, 1]
    k0, k1 = spline.domain
    pts = (pts - k0) / (k1 - k0)

    # the first basis function is always the 0
    first_basis_ids = np.zeros(pts.size, dtype=np.int_)

    return evaluate_Bernstein_basis(spline.degree, pts), first_basis_ids


def evaluate_Bspline_basis(
    spline: "Bspline1D", pts: FloatLike_32_64
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int32]]:
    """Evaluate B-spline basis functions at given points.

    This function automatically selects the most efficient evaluation method:
    - For Bézier-like knots: direct Bernstein evaluation
    - For general knots: Cox-de Boor recursion

    In both cases it calls vectorized or numba implementations.

    Args:
        spline (Bspline1D): B-spline object defining the basis.
        pts (FloatLike_32_64): Evaluation points.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.int32]]: Tuple of
            (basis_values, first_basis_indices) where basis_values is an array of
            shape (number pts, degree+1) that contains the basis function values at each point
            and first_basis_indices contains the index of the first non-zero basis
            function for each point.

    Raises:
        ValueError: If any evaluation points are outside the B-spline domain.

    Example:
        >>> bspline = Bspline1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
        >>> evaluate_Bspline_basis(bspline, [0.0, 0.5, 0.75, 1.0])
        (array([[1.        , 0.        , 0.        ],
                [0.12698413, 0.5643739 , 0.30864198],
                [0.69444444, 0.27777778, 0.02777778],
                [0.        , 0.        , 1.        ]]),
         array([0, 1, 3, 3]))
    """
    pts = _prepare_pts_for_evaluation(pts)
    pts = pts.astype(spline.dtype)

    if not np.all(
        is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)
    ):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    if spline.has_Bezier_like_knots():
        return _evaluate_Bspline_basis_Bernstein_like(spline, pts)
    else:
        return evaluate_basis_Cox_de_Boor_impl(
            spline.knots, spline.degree, spline.periodic, spline.tolerance, pts
        )
