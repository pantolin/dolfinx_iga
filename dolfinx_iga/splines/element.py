"""Custom basix element for splines."""

from __future__ import annotations

import typing

import basix
import basix.ufl
import numpy as np
from basix import CellType

from ..splines.basis import evaluate_cardinal_Bspline_basis

FiniteElementBase = basix.ufl._ElementBase

if typing.TYPE_CHECKING:
    from typing import Optional


def _get_cell_type(dim: int) -> CellType:
    """Get the cell type for a given dimension."""
    assert dim in [1, 2, 3], "Only  1D, 2D, and 3D dims are supported"
    return [CellType.interval, CellType.quadrilateral, CellType.hexahedron][dim - 1]


def _get_num_vertices(dim: int) -> int:
    """Get the number of vertices for a given dimension."""
    assert dim in [1, 2, 3], "Only  1D, 2D, and 3D dims are supported"
    return [2, 4, 8][dim - 1]


def _get_num_edges(dim: int) -> int:
    """Get the number of edges for a given dimension."""
    assert dim in [1, 2, 3], "Only  1D, 2D, and 3D dims are supported"
    return [0, 4, 12][dim - 1]


def _get_num_faces(dim: int) -> int:
    """Get the number of faces for a given dimension."""
    assert dim in [1, 2, 3], "Only  1D, 2D, and 3D dims are supported"
    return [0, 0, 6][dim - 1]


def _create_dolfinx_custom_element_int_points(
    degrees: list[int],
    dtype: np.dtype,
) -> list[list[np.ndarray]]:
    """Create the integration points for the dolfinx custom element.

    Create the integration points for the dolfinx custom element.
    This function creates the integration points for the dolfinx custom element,
    by creating the integration points for each dimension.

    No points are associated with vertices, edges, and faces.
    Only the integration points for the domain are created.

    Args:
        degrees (list[int]): The degrees of the basis functions.
        dtype (np.dtype): The data type of the integration points.

    Returns:
        list[list[np.ndarray]]: The integration points for the dolfinx custom element.
    """

    dim = len(degrees)

    x = [[], [], [], []]

    zeros = np.zeros((0, dim), dtype=dtype)
    x[0].extend([zeros] * _get_num_vertices(dim))
    x[1].extend([zeros] * _get_num_edges(dim))
    x[2].extend([zeros] * _get_num_faces(dim))

    xint_1D = []
    interval = _get_cell_type(1)
    for deg in degrees:
        n_pts_dir = deg + 1
        degree_of_exactness = 2 * n_pts_dir - 1
        xint_1D.append(basix.make_quadrature(interval, degree_of_exactness)[0])

    grids = np.array(np.meshgrid(*xint_1D, indexing="ij"))
    xint = grids.reshape(dim, -1).T
    # xint has C ordering (y coordinates run faster than x coordinates)

    x[dim].append(xint.astype(dtype).copy())

    return x


def _create_dolfinx_custom_element_wcoeffs(
    degrees: list[int], dtype: np.dtype
) -> np.ndarray:
    """Create the basis coefficients for the dolfinx custom element.

    This corresponds to the B matrix in the basix documentation
    (see https://docs.fenicsproject.org/basix/v0.9.0/cpp/classbasix_1_1FiniteElement.html#a5f8794de82cfc63ce8e40fad99802dfe)

    Args:
        degrees (list[int]): The degrees of the basis functions.
        dtype (np.dtype): The data type of the basis coefficients.

    Returns:
        np.ndarray: The basis coefficients for the dolfinx custom element.
    """
    dim = len(degrees)
    assert dim in [1, 2, 3], "Only 1D, 2D, and 3D splines are supported"

    supdegree = np.max(degrees)

    n_basis_dir = np.array(degrees) + 1
    n_sup_basis_dir = np.array([supdegree + 1] * dim)
    n_basis = np.prod(n_basis_dir)
    n_sup_basis = np.prod(n_sup_basis_dir)

    wcoeffs = np.eye(n_sup_basis, dtype=dtype)

    if n_basis == n_sup_basis:
        return wcoeffs  # all basis are active

    # Otherwise, only the active basis in the superbasis are selected

    if dim == 1:
        return wcoeffs[:n_basis]

    # Creating an array of multi-indices (only for the active bases in the superbasis).
    shape = tuple(degree + 1 for degree in degrees)
    multiindices = (
        np.array(np.meshgrid(*[range(s) for s in shape], indexing="ij"))
        .reshape(len(degrees), -1)
        .T
    )

    active_bases = [np.ravel_multi_index(i, n_sup_basis_dir) for i in multiindices]

    return wcoeffs[active_bases, :]


def _create_dolfinx_custom_element_CT(
    degrees: list[int], dtype: np.dtype
) -> np.ndarray:
    """Create the change of basis matrix from the Legendre polynomials
    to the cardinal B-spline basis.

    This matrix is such that:
        [cardinal values] = C @ [Legendre values]
    where:
    - legendre is the Legendre polynomial basis (rows) evaluated at the points (columns)
    - cardinal is the cardinal B-spline basis (rows) evaluated at the points (columns)

    This method returns the transpose of C (that is computed as a L2 projection of the cardinal basis into the Legendre basis)

    Args:
        degrees (list[int]): The degrees of the basis functions.
        dtype (np.dtype): The data type of the basis coefficients.

    Returns:
        np.ndarray: The C^T matrix (transpose of C) for the dolfinx custom element.
    """

    dim = len(degrees)
    sup_degree = np.max(degrees)
    cell_type = _get_cell_type(dim)

    # Quadrature is precise enough to integrate the product of
    # two polynomials of degree sup_degree.
    prod_degree = 2 * sup_degree
    quad_pts, quad_wts = basix.make_quadrature(cell_type, prod_degree)

    # Polynomial basis (Legendre, orthonormal).
    # The rows of the array are basis, and the columns are the points.
    legendre = basix.tabulate_polynomials(
        basix.PolynomialType.legendre, cell_type, sup_degree, quad_pts
    ).astype(dtype)

    quad_pts = quad_pts.astype(dtype)
    quad_wts = quad_wts.astype(dtype)

    # Here the rows are the points, and the columns are the basis.
    splines = evaluate_cardinal_Bspline_basis(degrees, quad_pts, funcs_order="F")

    # This is just a L2 projection of the cardinal basis into the polynomial basis.
    # As the polynomial basis is orthonormal, the mass matrix is the identity.
    # Thus, we just need to compute the right-hand-side integral
    # CT_ij = \int legendre_i(x) splines_j(x) dx
    CT = (legendre * quad_wts) @ splines
    return CT


def _create_dolfinx_custom_element_M(degrees: list[int], pts: np.ndarray) -> np.ndarray:
    """Create the M matrix for the dolfinx custom element.
    See https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.create_custom_element
    and https://docs.fenicsproject.org/basix/main/cpp/classbasix_1_1FiniteElement.html#a5f8794de82cfc63ce8e40fad99802dfe

    The M matrix is computed only for the interior of the domain,
    not for the vertices, edges, and faces.

    Args:
        degrees (list[int]): The degrees of the basis functions.
        pts (np.ndarray): The integration points in the interior of the domain.

    Returns:
        np.ndarray: The M matrix for the dolfinx custom element.

    """

    dim = len(degrees)
    n_basis = np.prod(np.array(degrees) + 1)

    n_pts = pts.shape[0]
    dtype = pts.dtype

    cell_type = _get_cell_type(dim)

    # For computing the M matrix we use the identity C @ D^T = I (see the second link),
    # where D = M @ P(pts)^T and P(pts) is the polynomial evaluation at the integration points,
    # and C is the change of basis matrix from the Legendre polynomials to the cardinal B-spline basis.
    # Therefore C @ P(pts) @ M^T = I and therefore M = (P(pts)^T @ C^T)^-1

    # These are (orthonormal) Legendre polynomials.
    # The rows of the array are basis, and the columns are the points.
    sup_degree = np.max(degrees)
    legendre = basix.tabulate_polynomials(
        basix.PolynomialType.legendre, cell_type, sup_degree, pts
    ).astype(dtype)

    # Transpose of the change of basis matrix from the Legendre polynomials to the cardinal B-spline basis.
    CT = _create_dolfinx_custom_element_CT(degrees, dtype)

    # This is the inverse of the change of basis matrix.
    Mint = np.linalg.inv(legendre.T @ CT)

    M = [[], [], [], []]
    zeros = np.zeros((0, 1, 0, 1), dtype=dtype)
    M[0].extend([zeros] * _get_num_vertices(dim))
    M[1].extend([zeros] * _get_num_edges(dim))
    M[2].extend([zeros] * _get_num_faces(dim))
    M[dim].append(Mint.reshape(n_basis, 1, n_pts, 1))

    return M


def _create_dolfinx_custom_element(
    degrees: list[int], dtype: np.dtype
) -> FiniteElementBase:
    """Create the dolfinx custom element.

    This function creates the dolfinx custom element,
    by creating the basis coefficients, the integration points, and the mass matrix.

    Args:
        degrees (list[int]): The degrees of the basis functions.
        dtype (np.dtype): The data type of the basis coefficients.

    Returns:
        FiniteElementBase: The dolfinx custom element.
    """
    dim = len(degrees)

    assert dim in [1, 2, 3], "Only 1D, 2D, and 3D splines are supported"

    cell_type = _get_cell_type(dim)

    wcoeffs = _create_dolfinx_custom_element_wcoeffs(degrees, dtype)
    x = _create_dolfinx_custom_element_int_points(degrees, dtype)
    M = _create_dolfinx_custom_element_M(degrees, x[dim][0])

    return basix.ufl.custom_element(
        cell_type=cell_type,
        reference_value_shape=[],
        wcoeffs=wcoeffs,
        x=x,
        M=M,
        interpolation_nderivs=0,
        map_type=basix.MapType.identity,
        sobolev_space=basix.SobolevSpace.H1,
        discontinuous=True,
        embedded_subdegree=np.min(degrees),
        embedded_superdegree=np.max(degrees),
        polyset_type=basix.PolysetType.standard,
        dtype=dtype,
    )


def create_cardinal_Bspline_element(
    degrees: list[int],
    shape: Optional[tuple[int, ...]] = None,
    symmetry: Optional[bool] = None,
    dtype: np.dtype = np.float64,
) -> basix.ufl._ElementBase:
    """Create a cardinal B-spline element.

    The dimension of the element is determined by the length of the degrees list.

    Args:
        degrees (list[int]): The degrees of the basis functions.
        shape (Optional[tuple[int, ...]]): The shape of the element.
        symmetry (Optional[bool]]): Whether the element is symmetric.
        dtype (np.dtype): The data type of the element. Defaults to np.float64.

    Returns:
        basix.ufl._ElementBase: The cardinal B-spline element.

    Raises:
        ValueError: If the degrees are not non-negative.
        ValueError: If the dimension is not 1, 2, or 3.
        ValueError: If the shape is not None and the symmetry is not None.
    """

    dim = len(degrees)

    if dim < 1 or dim > 3:
        raise ValueError("Only 1D, 2D, and 3D splines are supported")

    if np.any(np.array(degrees) < 0):
        raise ValueError("Degree must be non-negative")

    ufl_e = _create_dolfinx_custom_element(degrees, dtype)

    if shape is None:
        if symmetry is not None:
            raise ValueError("Cannot pass a symmetry argument to this element.")
        return ufl_e
    else:
        return basix.ufl.blocked_element(ufl_e, shape=shape, symmetry=symmetry)
