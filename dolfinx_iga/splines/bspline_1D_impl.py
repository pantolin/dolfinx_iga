"""Implementation functions for 1D B-spline space operations.

This module provides low-level, Numba-accelerated implementations of B-spline
operations including basis function evaluation, knot analysis, and geometric
computations.
"""

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.types import Tuple as nb_Tuple
from numba.types import bool as nb_bool
from numba.types import float32, float64, intp, void

from ..utils.types import Float_32_64, FloatArray_32_64, IntArray


@njit(
    [
        void(float32[::1], intp),
        void(float64[::1], intp),
    ],
    cache=True,
)
def _assert_spline_info(knots: FloatArray_32_64, degree: np.int_) -> None:
    """Assert basic B-spline validity conditions.

    The idea of this function is to raise exceptions during development,
    and remove them in production (e.g., with `__debug__` set to `False`,
    or by executing the code with `python -O`).

    Args:
        knots (FloatArray_32_64): B-spline knot vector to validate.
        degree (np.int_): Degree to validate.

    Raises:
        AssertionError: If knots is not 1D, degree is negative, knots
            has insufficient elements, or knots are not non-decreasing.
    """
    assert knots.ndim == 1, "knots must be a 1D array"
    assert degree >= 0, "degree must be non-negative"
    assert knots.size >= (2 * degree + 2), (
        "knots must have at least 2*degree+2 elements"
    )
    assert np.all(np.diff(knots) >= knots.dtype.type(0.0)), (
        "knots must be non-decreasing"
    )


@njit(
    [
        intp(float32[::1], intp, float32),
        intp(float64[::1], intp, float64),
    ],
    cache=True,
)
def get_multiplicity_of_first_knot_in_domain_impl(
    knots: FloatArray_32_64,
    degree: np.int_,
    tol: Float_32_64,
) -> np.int_:
    """Get the multiplicity of the first knot in the domain (i.e.,
    the `degree`-th knot).

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        tol (Float_32_64): Tolerance for numerical comparisons.

    Returns:
        np.int_: Multiplicity of the first knot in the domain.

    Raises:
        AssertionError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    if __debug__:
        _assert_spline_info(knots, degree)
        assert tol > 0, "tol must be positive"

    first_knot = knots[degree]
    return np.int_(np.sum(np.isclose(knots[: degree + 1], first_knot, atol=tol)))


@njit(
    [
        nb_Tuple((float32[:], intp[:]))(float32[::1], intp, float32, nb_bool),
        nb_Tuple((float64[:], intp[:]))(float64[::1], intp, float64, nb_bool),
    ],
    cache=True,
)
def get_unique_knots_and_multiplicity_impl(
    knots: FloatArray_32_64,
    degree: np.int_,
    tol: Float_32_64,
    in_domain: bool = False,
) -> tuple[FloatArray_32_64, IntArray]:
    """Get unique knots and their multiplicities.

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        tol (Float_32_64): Tolerance for numerical comparisons.
        in_domain (bool): If True, only consider knots in the domain.
            Defaults to False.

    Returns:
        tuple[FloatArray_32_64, IntArray]: Tuple of (unique_knots, multiplicities)
            where unique_knots contains the distinct knot values and multiplicities
            contains the corresponding multiplicity counts. Both arrays have
            the same length.

    Raises:
        AssertionError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    if __debug__:
        _assert_spline_info(knots, degree)
        assert tol > 0, "tol must be positive"

    # Round to tolerance precision for grouping
    dtype = knots.dtype
    scale = dtype.type(1.0 / tol)
    rounded_knots = np.round(knots * scale) / scale

    n = knots.size
    # unique_rounded_knots = np.empty(n, dtype=rounded_knots.dtype)
    unique_rounded_knots_ids = np.empty(n, dtype=np.int_)
    mult = np.zeros(n, dtype=np.int_)

    if in_domain:
        rknot_0, rknot_1 = rounded_knots[degree], rounded_knots[-degree - 1]
    else:
        rknot_0, rknot_1 = rounded_knots[0], rounded_knots[-1]

    j = -1
    last_rknot = np.nan

    for i, rknot in enumerate(rounded_knots):
        if rknot < rknot_0:
            continue
        elif rknot > rknot_1:
            break

        if rknot == last_rknot:
            mult[j] += 1
        else:
            j += 1
            last_rknot = rknot
            unique_rounded_knots_ids[j] = i
            mult[j] = 1

    unique_knots = rounded_knots[unique_rounded_knots_ids[: j + 1]]
    mults = mult[: j + 1]
    return unique_knots, mults


@njit(
    [
        nb_bool[:](float32[::1], intp, float32[::1], float32),
        nb_bool[:](float64[::1], intp, float64[::1], float64),
    ],
    cache=True,
)
def is_in_domain_impl(
    knots: FloatArray_32_64,
    degree: np.int_,
    pts: FloatArray_32_64,
    tol: Float_32_64,
) -> npt.NDArray[np.bool_]:
    """Check if points are within the B-spline domain (up to tolerance).

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        pts (FloatArray_32_64): Points to check.
        tol (Float_32_64): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates points
            are within the domain. It has the same length as the number of points.

    Raises:
        AssertionError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    pts_arr = np.asarray(pts, dtype=knots.dtype)

    if __debug__:
        _assert_spline_info(knots, degree)
        assert tol > 0, "tol must be positive"
        assert pts_arr.ndim == 1, "pts must be a 1D array"
        assert pts_arr.size > 0, "pts must have at least one element"

    knot_begin, knot_end = knots[degree], knots[-degree - 1]
    return np.logical_and(
        (knot_begin < pts_arr) | np.isclose(knot_begin, pts_arr, atol=tol),
        (pts_arr < knot_end) | np.isclose(pts_arr, knot_end, atol=tol),
    )


@njit(
    [
        intp(float32[::1], intp, nb_bool, float32),
        intp(float64[::1], intp, nb_bool, float64),
    ],
    cache=True,
)
def compute_num_basis_impl(
    knots: FloatArray_32_64,
    degree: np.int_,
    periodic: bool,
    tol: Float_32_64,
) -> np.int_:
    """Compute the number of basis functions.

    In the non-periodic case, the number of basis functions is given by
    the number of knots minus the degree minus 1.

    In the periodic case, the number of basis functions is computed as
    before, minus the regularity at the domain's beginning/end minus 1.

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (Float_32_64): Tolerance for numerical comparisons.

    Returns:
        np.int_: Number of basis functions.

    Raises:
        AssertionError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """

    if __debug__:
        _assert_spline_info(knots, degree)
        assert tol > 0, "tol must be positive"

    num_basis = np.int_(len(knots) - degree - 1)

    if periodic:
        # Determining the number of extra basis required in the periodic case.
        # This depends on the regularity of the knot vector at domain's
        # begining.
        regularity = degree - get_multiplicity_of_first_knot_in_domain_impl(
            knots, degree, tol
        )
        num_basis -= regularity + 1

    return num_basis


@njit(
    [
        intp[:](float32[::1], float32[:]),
        intp[:](float64[::1], float64[:]),
    ],
    cache=True,
)
def get_last_knot_smaller_equal_impl(
    knots: FloatArray_32_64,
    pts: FloatArray_32_64,
) -> npt.NDArray[np.int_]:
    """Get the index of the last knot which is less than or equal to each point in pts.

    Args:
        knots (FloatArray_32_64): B-spline knot vector (must be non-decreasing).
        pts (FloatArray_32_64): Points (1D array) to find knot indices for.

    Returns:
        npt.NDArray[np.int_]: Array of computed indices, one for each point in pts.

    Raises:
        AssertionError: If knots are not non-decreasing, or is not a 1D array,
            or points array is invalid (is not a 1D array or has no elements).
    """

    if __debug__:
        assert knots.ndim == 1, "knots must be a 1D array"
        assert np.all(np.diff(knots) >= knots.dtype.type(0.0)), (
            "knots must be non-decreasing"
        )
        assert pts.ndim == 1, "pts must be a 1D array"
        assert pts.size > 0, "pts must have at least one element"

    return np.searchsorted(knots, pts, side="right") - 1


@njit(
    [
        nb_Tuple((float32[:, ::1], intp[:]))(
            float32[::1], intp, nb_bool, float32, float32[::1]
        ),
        nb_Tuple((float64[:, ::1], intp[:]))(
            float64[::1], intp, nb_bool, float64, float64[::1]
        ),
    ],
    cache=True,
)
def evaluate_basis_Cox_de_Boor_impl(
    knots: FloatArray_32_64,
    degree: np.int_,
    periodic: bool,
    tol: Float_32_64,
    pts: FloatArray_32_64,
) -> tuple[FloatArray_32_64, IntArray]:
    """Evaluate B-spline basis functions using Cox-de Boor recursion.

    This function implements Algorithm 2.23 from "Spline Methods Draft" by Tom Lyche.

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (Float_32_64): Tolerance for numerical comparisons.
        pts (FloatArray_32_64): Points (1D array) to evaluate basis functions at.

    Returns:
        tuple[FloatArray_32_64, IntArray]: Tuple of (basis_values, first_basis_indices)
            where basis_values contains the basis function values at each point
            and first_basis_indices contains the index of the first non-zero
            basis function for each point.

    Raises:
        AssertionError: If tolerance is not positive, basic validation for knots, degree, and pts fails, or points are outside domain.
    """
    # See Spline Methods Draft, by Tom Lychee. Algorithm 2.23

    if __debug__:
        _assert_spline_info(knots, degree)
        assert tol > 0, "tol must be positive"
        assert pts.ndim == 1, "pts must be a 1D array"
        assert pts.size > 0, "pts must have at least one element"

    knot_ids = get_last_knot_smaller_equal_impl(knots, pts)

    assert np.all(is_in_domain_impl(knots, degree, pts, tol)) and np.all(
        knot_ids <= (knots.size - 1)
    )

    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    order = degree + 1
    n_pts = pts.size

    basis = np.zeros((n_pts, order), dtype=dtype)
    basis[:, -1] = one

    # Here we account for the case where the evaluation point
    # coincides with the last knot.
    num_basis = compute_num_basis_impl(knots, degree, periodic, tol)
    first_basis = np.minimum(knot_ids - degree, num_basis - order)

    for pt_id in range(n_pts):
        knot_id = knot_ids[pt_id]

        if knot_id == (knots.size - 1):
            continue

        pt = pts[pt_id]
        basis_i = basis[pt_id, :]
        local_knots = knots[knot_id - degree + 1 : knot_id + order]

        for sub_degree in range(1, order):
            k0, k1 = local_knots[0], local_knots[sub_degree]
            diff = k1 - k0
            inv_diff = zero if diff < tol else one / diff

            for bs_id in range(degree - sub_degree, degree):
                basis_i[bs_id] *= (pt - k0) * inv_diff

                k0, k1 = local_knots[bs_id], local_knots[bs_id + sub_degree]
                diff = k1 - k0
                inv_diff = zero if diff < tol else one / diff

                basis_i[bs_id] += (k1 - pt) * inv_diff * basis_i[bs_id + 1]

            basis_i[-1] *= (pt - k0) * inv_diff

    return basis, first_basis


@njit(
    [
        nb_bool[:](float32[::1], intp, float32),
        nb_bool[:](float64[::1], intp, float64),
    ],
    cache=True,
)
def get_cardinal_intervals_impl(
    knots: FloatArray_32_64, degree: np.int_, tol: Float_32_64
) -> npt.NDArray[np.bool_]:
    """Get boolean array indicating whether intervals are cardinal.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        tol (Float_32_64): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates cardinal intervals.
            It has length equal to the number of intervals.

    Raises:
        AssertionError: If tolerance is not positive or basic validation of knots and degree fails.
    """
    _, mult = get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    cardinal = np.full(num_intervals, np.False_, dtype=np.bool_)

    if np.all(mult > 1):
        return cardinal

    knot_id = degree

    # Note: this loop could be shortened by only looking at those
    # intervals for which the multiplicity of the first knot is 1.
    # This would require to compute knot_id differently.
    for elem_id in range(num_intervals):
        if mult[elem_id] == 1 and mult[elem_id + 1] == 1:
            local_knots = knots[knot_id - degree + 1 : knot_id + degree + 1]
            lengths = np.diff(local_knots)
            if np.all(np.isclose(lengths, lengths[degree - 1], atol=tol)):
                cardinal[elem_id] = np.True_

        knot_id += mult[elem_id + 1]

    return cardinal


@njit(
    [
        float32[:, :, ::1](float32[::1], intp, float32),
        float64[:, :, ::1](float64[::1], intp, float64),
    ],
    cache=True,
)
def create_bspline_Bezier_extraction_operators_impl(
    knots: FloatArray_32_64, degree: np.int_, tol: Float_32_64
) -> FloatArray_32_64:
    """Create Bézier extraction operators for each interval.

    This function computes the extraction operators that transform Bernstein
    into B-spline basis functions for each interval.
    For each interval \( i \), the Bézier extraction operator \( C_i \) satisfies:

        \[
        N_i(x) = C_i @ B(ξ)
        \]

    where:
      - N_i(x) is the vector of B-spline basis functions nonzero on the interval \( i \), evaluated at \( x \),
      - \( B(ξ) \) is the vector of Bernstein basis functions on the reference interval \([0, 1]\), evaluated at \( ξ \),
      - \( C_i \) is the extraction matrix for interval \( i \),
      - \( x \) is the physical coordinate, \( ξ \) is the local (reference) referred to \([0, 1]\).

    Args:
        knots (FloatArray_32_64): B-spline knot vector.
        degree (np.int_): B-spline degree.
        tol (Float_32_64): Tolerance for numerical comparisons.

    Returns:
        FloatArray_32_64: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            Bernstein basis functions to B-spline basis functions for that interval.

    Raises:
        AssertionError: If tolerance is not positive or basic validation of knots and degree fails.
    """
    unique_knots, mults = get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=True
    )

    if __debug__:
        _assert_spline_info(knots, degree)
        assert tol > 0, "tol must be positive"

    n_elems = len(unique_knots) - 1

    dtype = knots.dtype
    one = dtype.type(1.0)

    # Initialize identity matrix for every element.
    Cs = np.zeros((n_elems, degree + 1, degree + 1), dtype=dtype)
    Cs[:, : degree + 1, : degree + 1] = np.eye(degree + 1, dtype=dtype)

    mult = get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)

    # If not open first knot, additional knot insertion is needed.
    if mult < (degree + 1):
        C = Cs[0]
        reg = degree - mult

        t = knots[degree]
        for r in range(reg):
            lcl_knots = knots[r:]
            for k in range(1, degree - r):
                alpha = (t - lcl_knots[k]) / (lcl_knots[k + degree - r] - lcl_knots[k])
                C[:, k - 1] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

    alphas = np.zeros(degree - 1, dtype=dtype)

    knt_id = degree
    mult = 0

    for elem_id in range(n_elems):
        knt_id += mult
        mult = mults[elem_id + 1]

        if mult >= degree:
            continue

        lcl_knots = knots[knt_id : knt_id + degree + 1]
        alphas[: degree - mult] = (lcl_knots[1] - lcl_knots[0]) / (
            lcl_knots[mult + 1 :] - lcl_knots[0]
        )

        C = Cs[elem_id]

        reg = degree - mult
        for r in range(1, reg + 1):
            s = mult + r
            for k in range(degree, s - 1, -1):
                alpha = alphas[k - s]
                C[:, k] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

            if elem_id < (n_elems - 1):
                Cs[elem_id + 1, reg - r : reg + 1, reg - r] = C[
                    degree - r : degree + 1, degree
                ]

    return Cs
