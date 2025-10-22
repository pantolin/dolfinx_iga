from typing import Callable, Optional

import numpy as np
from basix import LagrangeVariant

from ..utils.types import FloatArray_32_64, FloatLike_32_64
from .basis_1D import (
    evaluate_Bernstein_basis_1D,
    evaluate_cardinal_Bspline_basis_1D,
    evaluate_Lagrange_basis_1D,
)


def _prepare_pts_for_evaluation(
    pts: FloatLike_32_64 | list[FloatArray_32_64],
    dim: int,
    pts_order: Optional[str] = None,
) -> FloatArray_32_64 | list[FloatArray_32_64]:
    """
    Standardizes input points for basis function evaluation in arbitrary dimension.

    This utility ensures that the provided points are correctly shaped and typed
    for evaluation routines, handling scalars, 1D/2D arrays, and lists of arrays
    as appropriate for the dimension. It converts scalars to arrays, handles common
    array and list input types, and checks consistency between the size of the
    points and the specified dimension.

    Args:
        pts (FloatLike_32_64 or list[FloatArray_32_64]): Physical points at which to evaluate the basis. This can be:
            - a scalar,
            - a 1D array of coordinates (for 1D),
            - a 2D array of shape (n_pts, dim),
            - or a list of length `dim`, where each entry is a 1D array of shape (n_pts,).
        dim (int): Geometric dimension of the points (e.g., 1, 2, 3).
        pts_order (str, optional): If points are provided as a tensor-product grid, this specifies the ordering:
            'C' for C-order (last index fastest), 'F' for Fortran-order (first index fastest).
            For general points, leave as None.

    Returns:
        FloatArray_32_64 or list[FloatArray_32_64]: Points rearranged/converted as list[FloatArray_32_64] of length `dim`,
            or unchanged if already in required list form.

    Raises:
        AssertionError: If `dim` is not positive.
        ValueError: If the list does not have `dim` elements, multidimensional arrays do not
            match `dim`, points are of unsupported dimensionality, or ordering requested
            when not meaningful.
    """

    assert dim > 0, "Invalid dimension."

    if isinstance(pts, list):
        if len(pts) != dim:
            raise ValueError(f"pts must have {dim} elements (the same as evaluators).")
        return pts

    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if np.issubdtype(pts.dtype, np.integer):
        pts = pts.astype(np.float64)

    if pts.ndim == 0:
        pts = np.array([pts], dtype=pts.dtype)
    elif pts.ndim == 2:
        if pts.shape[1] == 1:
            pts = pts.flatten()
        elif pts.shape[1] != dim:
            raise ValueError(f"pts must have {dim} rows")
        elif pts_order is not None:
            raise ValueError(
                "If multidimensional points are provided, pts_order must be None."
            )
    elif pts.ndim > 2:
        raise ValueError("pts must be either 1D or 2D array")

    if pts.ndim == 1:
        pts = [pts] * dim

    if pts_order is not None and pts_order not in ("F", "C"):
        raise ValueError("Invalid points order.")

    return pts


def _extract_points_tensor_grid(
    pts: FloatArray_32_64,
) -> tuple[list[FloatArray_32_64], str] | None:
    """
    Extracts 1D coordinate arrays that define a tensor-product grid from a 2D array of points,
    and determines the meshgrid order (row-major/"C", column-major/"F") if possible.

    This utility examines a 2D array of points of shape (n_points, dim),
    checks if the points form a regular tensor grid (meshgrid) with unique,
    sorted 1D coordinates in each dimension, and determines whether the memory order
    of the (flattened) meshgrid matches "C" (last index varies fastest) or "F" (first index varies fastest)
    conventions. The returned list contains the sorted unique coordinates for each dimension,
    intended for 1D basis function evaluation. The function can be used to efficiently
    exploit tensor-product structure for evaluating basis or quadrature on structured grids.

    Args:
        pts (FloatArray_32_64): Array with shape (n_points, dim).
            Each row represents a point in `dim`-dimensional space.

    Returns:
        tuple[list[FloatArray_32_64], str] | None:
            - Tuple: ([x_0, ..., x_{dim-1}], order), where x_i is np.ndarray of unique
              grid coordinates in dimension i, and order is "C" or "F" corresponding to meshgrid
              ordering ("xy" or "ij").
            - None if `pts` is not a valid tensor grid or does not match any known order.

    Example:
        For the following input in 2D:
            pts = array([[0.0, 2.0], [1.0, 2.0], [0.0, 3.0], [1.0, 3.0]])
        The function would return:
            ([array([0., 1.]), array([2., 3.])], "F")

    Raises:
        AssertionError: If `pts` is not 2D.
    """
    assert pts.ndim == 2

    dim = pts.shape[1]

    # For each dimension, get unique sorted entries:
    coords_1d = [np.unique(pts[:, i]) for i in range(dim)]

    # Check if n_pts is the product of the unique coordinate counts
    n_pts = np.prod([len(c) for c in coords_1d])
    if n_pts == pts.shape[0]:
        # Try to reconstruct the meshgrid and see if matches with either C or F order
        mg_c = np.array(np.meshgrid(*coords_1d, indexing="ij")).reshape(dim, -1).T
        if np.allclose(mg_c, pts):
            return coords_1d, "C"  # C-order (last coordinate runs faster)

        mg_f = np.array(np.meshgrid(*coords_1d, indexing="xy")).reshape(dim, -1).T
        if np.allclose(mg_f, pts):
            return coords_1d, "F"  # Fortran-order (first coordinate runs faster)
        # Otherwise, tensor-grid-like but scrambled; fall through

    return None


def _basis_combinator_tensor_grid(
    evaluators_1D: list[Callable[[FloatLike_32_64], FloatArray_32_64]],
    pts: list[FloatArray_32_64],
    funcs_order: str,
    pts_order: Optional[str] = None,
) -> FloatArray_32_64:
    """Combine 1D basis functions evaluated at a tensor-product grid of points.
    This function efficiently evaluates a set of 1D basis functions at a tensor-product grid
    of points, exploiting the structured nature of the grid to minimize redundant computation.
    It handles both C-order (last index varies fastest) and F-order (first index varies fastest)
    meshgrid layouts, and correctly combines the results according to the specified function
    and point orderings.

    Args:
        evaluators_1D (list[Callable[[FloatLike_32_64], FloatArray_32_64]]): List of 1D basis function evaluators,
            each taking a 1D array of points and returning an array of basis function values.
        pts (list[FloatArray_32_64]): List of 1D arrays of points, one for each dimension.
        funcs_order (str): Ordering of the basis functions: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'F'.
        pts_order (str, optional): Ordering of the points: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to None, which means 'F'. Defaults to None. If not provided, it will be set to 'F'.

    Returns:
        FloatArray_32_64: Array of shape (n_points, n_basis_functions) containing the combined basis function values.

    Raises:
        AssertionError: If the dimension is invalid, the number of points is not equal to the number of evaluators,
        the functions order is invalid, or the points order is invalid.
    """
    dim = len(evaluators_1D)
    if pts_order is None:
        pts_order = "F"

    assert dim > 0, "Invalid dimension."
    assert len(pts) == dim, (
        "The number of points must be equal to the number of evaluators."
    )
    assert funcs_order in ("F", "C"), "Invalid functions order."
    assert pts_order in ("F", "C"), "Invalid points order."

    pts_order_str = "qp" if pts_order == "F" else "pq"
    funcs_order_str = "ji" if funcs_order == "F" else "ij"

    op_str = f"pi,qj->{pts_order_str}{funcs_order_str}"

    out = evaluators_1D[0](pts[0])
    for dir in range(1, dim):
        vals_1D = evaluators_1D[dir](pts[dir])
        n_rows = out.shape[0] * vals_1D.shape[0]
        n_cols = out.shape[1] * vals_1D.shape[1]
        out = np.einsum(op_str, out, vals_1D).reshape(n_rows, n_cols)

    return out


def _basis_combinator_array(
    evaluators_1D: list[Callable[[FloatLike_32_64], FloatArray_32_64]],
    pts: FloatLike_32_64,
    funcs_order: str = "F",
) -> FloatArray_32_64:
    """
    Combine 1D basis functions evaluated at a collection of points.
    This function efficiently evaluates a set of 1D basis functions at a collection of points (not a tensor-product grid),
    by iterating over the dimensions and combining the results according to the specified function
    and point orderings.

    Args:
        evaluators_1D (list[Callable[[FloatLike_32_64], FloatArray_32_64]]): List of 1D basis function evaluators,
            each taking a 1D array of points and returning an array of basis function values.
        pts (FloatLike_32_64): Points to evaluate the basis functions at.
        funcs_order (str): Ordering of the basis functions: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'F'.

    Returns:
        FloatArray_32_64: Array of shape (n_points, n_basis_functions) containing the combined basis function values.

    Raises:
        AssertionError: If the points are not 2D.
    """
    dim = pts.shape[1]

    # Checking if pts has a tensor-product distribution.
    pts_out = _extract_points_tensor_grid(pts)
    if pts_out is not None:
        pts, pts_order = pts_out
        return _basis_combinator_tensor_grid(evaluators_1D, pts, funcs_order, pts_order)

    funcs_order_str = "ji" if funcs_order == "F" else "ij"
    op_str = f"pi,pj->p{funcs_order_str}"
    n_pts = pts.shape[0]

    out = evaluators_1D[0](pts[:, 0])
    for dir in range(1, dim):
        vals_1D = evaluators_1D[dir](pts[:, dir])
        out = np.einsum(op_str, out, vals_1D).reshape(n_pts, -1)

    return out


def basis_1D_combinator(
    evaluators_1D,
    pts: FloatLike_32_64 | list[FloatArray_32_64],
    funcs_order: str = "F",
    pts_order: Optional[str] = None,
) -> FloatArray_32_64:
    """
    Efficiently combine 1D basis function evaluations into a multidimensional tensor-product basis.

    Evaluates and combines a list of 1D basis function evaluators at the given points, supporting both general arrays
    of points and tensor-product grids. Respects specified function and point ordering conventions, and dispatches
    to the optimal combination method depending on input shape. Handles both array input (each point is a row)
    and list-of-arrays input (tensor-product grids).

    Args:
        evaluators_1D (list[Callable[[FloatLike_32_64], FloatArray_32_64]]): List of 1D basis function evaluators,
            each taking a 1D array of points and returning an array of basis function values.
        pts (FloatLike_32_64 | list[FloatArray_32_64]): Points to evaluate the basis functions at.
        funcs_order (str): Ordering of the basis functions: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'F'.
        pts_order (str, optional): Ordering of the points: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to None, which means 'F'. Defaults to None. If not provided, it will be set to 'F'.

    Returns:
        FloatArray_32_64: Array of shape (n_points, n_basis_functions) containing the combined basis function values.

    Raises:
        ValueError: If the dimension is invalid, the functions order is invalid, or the points order is invalid.
        AssertionError: If the points are not 2D.
    """
    dim = len(evaluators_1D)

    if dim < 1:
        raise ValueError("Invalid dimension.")

    if funcs_order not in ("F", "C"):
        raise ValueError("Invalid functions order. It must be 'F' or 'C'.")
    if pts_order is not None and pts_order not in ("F", "C"):
        raise ValueError("Invalid points order. It must be 'F' or 'C'.")

    pts = _prepare_pts_for_evaluation(pts, dim, pts_order)

    if isinstance(pts, list):
        return _basis_combinator_tensor_grid(evaluators_1D, pts, funcs_order, pts_order)
    else:
        assert pts_order is None, "pts_order must be None when pts is not a list."
        return _basis_combinator_array(evaluators_1D, pts, funcs_order)


def evaluate_cardinal_Bspline_basis(
    degrees: list[int],
    pts: FloatLike_32_64 | list[FloatArray_32_64],
    funcs_order: str = "F",
    pts_order: Optional[str] = None,
) -> FloatArray_32_64:
    """Evaluate the cardinal B-spline basis functions at the given points.

    Evaluates cardinal B-spline basis functions by combining 1D basis values across each dimension,
    supporting both general points (e.g., a 2D array of shape (n_pts, dim) for scattered points)
    and tensor-product grids (a list of arrays, one for each coordinate direction). Fully supports
    C/F-ordering for functions and points. See `pts` argument for all accepted input formats.

    Args:
        degrees (list[int]): List of degrees of the B-spline basis functions.
        pts (FloatLike_32_64 | list[FloatArray_32_64]): Physical points at which to evaluate the basis.
            Acceptable forms:
              - scalar (for 1D),
              - 1D array of coordinates (for 1D),
              - 2D array of shape (n_points, dim) for scattered points,
              - list of length `dim`, each entry a 1D array (for tensor-product grid in each dimension).
        funcs_order (str): Ordering of the basis functions: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'F'.
        pts_order (str, optional): Ordering of the points: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to None, which means 'F'. Defaults to None. If not provided, it will be set to 'F'.

    Returns:
        FloatArray_32_64: Array of shape (n_points, n_basis_functions) containing the combined basis function values.

    Raises:
        ValueError: If the degrees are not non-negative, the functions order is invalid, or the points order is invalid.
    """

    if np.min(degrees) < 0:
        raise ValueError("All degrees must be non-negative")

    evaluators_1D = [
        lambda pts, d=degree: evaluate_cardinal_Bspline_basis_1D(d, pts)
        for degree in degrees
    ]
    return basis_1D_combinator(evaluators_1D, pts, funcs_order, pts_order)


def evaluate_Bernstein_basis(
    degrees: list[int],
    pts: FloatLike_32_64 | list[FloatArray_32_64],
    funcs_order: str = "F",
    pts_order: Optional[str] = None,
) -> FloatArray_32_64:
    """Evaluate the Bernstein basis functions at the given points.

    Evaluates Bernstein basis functions by combining 1D basis values across each dimension,
    supporting both general points (e.g., a 2D array of shape (n_pts, dim) for scattered points)
    and tensor-product grids (a list of arrays, one for each coordinate direction). Fully supports
    C/F-ordering for functions and points. See `pts` argument for all accepted input formats.

    Args:
        degrees (list[int]): List of degrees of the B-spline basis functions.
        pts (FloatLike_32_64 | list[FloatArray_32_64]): Physical points at which to evaluate the basis.
            Acceptable forms:
              - scalar (for 1D),
              - 1D array of coordinates (for 1D),
              - 2D array of shape (n_points, dim) for scattered points,
              - list of length `dim`, each entry a 1D array (for tensor-product grid in each dimension).
        funcs_order (str): Ordering of the basis functions: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'F'.
        pts_order (str, optional): Ordering of the points: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to None, which means 'F'. Defaults to None. If not provided, it will be set to 'F'.

    Returns:
        FloatArray_32_64: Array of shape (n_points, n_basis_functions) containing the combined basis function values.

    Raises:
        ValueError: If the degrees are not non-negative, the functions order is invalid, or the points order is invalid.
    """

    if np.min(degrees) < 0:
        raise ValueError("All degrees must be non-negative")

    evaluators_1D = [
        lambda pts, d=degree: evaluate_Bernstein_basis_1D(d, pts) for degree in degrees
    ]
    return basis_1D_combinator(evaluators_1D, pts, funcs_order, pts_order)


def evaluate_Lagrange_basis(
    degrees: list[int],
    pts: FloatLike_32_64 | list[FloatArray_32_64],
    lagrange_variant: LagrangeVariant = LagrangeVariant.equispaced,
    funcs_order: str = "F",
    pts_order: Optional[str] = None,
) -> FloatArray_32_64:
    """Evaluate the Lagrange basis functions at the given points.

    Evaluates Lagrange basis functions by combining 1D basis values across each dimension,
    supporting both general points (e.g., a 2D array of shape (n_pts, dim) for scattered points)
    and tensor-product grids (a list of arrays, one for each coordinate direction). Fully supports
    C/F-ordering for functions and points. See `pts` argument for all accepted input formats.

    Args:
        degrees (list[int]): List of degrees of the B-spline basis functions.
        pts (FloatLike_32_64 | list[FloatArray_32_64]): Physical points at which to evaluate the basis.
            Acceptable forms:
              - scalar (for 1D),
              - 1D array of coordinates (for 1D),
              - 2D array of shape (n_points, dim) for scattered points,
              - list of length `dim`, each entry a 1D array (for tensor-product grid in each dimension).
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, GLL, etc). Defaults to LagrangeVariant.equispaced.
        funcs_order (str): Ordering of the basis functions: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'F'.
        pts_order (str, optional): Ordering of the points: 'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to None, which means 'F'. Defaults to None. If not provided, it will be set to 'F'.

    Returns:
        FloatArray_32_64: Array of shape (n_points, n_basis_functions) containing the combined basis function values.

    Raises:
        ValueError: If the degrees are not non-negative, the functions order is invalid, or the points order is invalid.
    """

    if np.min(degrees) < 1:
        raise ValueError("All Lagrange basis degrees must be at least 1")

    evaluators_1D = [
        lambda pts, d=degree, variant=lagrange_variant: evaluate_Lagrange_basis_1D(
            d, pts, variant
        )
        for degree in degrees
    ]
    return basis_1D_combinator(evaluators_1D, pts, funcs_order, pts_order)
