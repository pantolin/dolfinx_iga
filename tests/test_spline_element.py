import basix
import numpy as np
import pytest
from basix import CellType

from dolfinx_iga.splines.basis_1D import evaluate_cardinal_Bspline_basis
from dolfinx_iga.splines.element import SplineElement


def test_spline_element_tabulate_vs_cardinal_basis():
    """Test that SplineElement.tabulate matches evaluate_cardinal_Bspline_basis."""
    degree = 3
    supdegree = degree + 5
    element = SplineElement(degree, supdegree)

    n_pts = 100
    eval_pts = basix.create_lattice(
        CellType.interval, n_pts, basix.LatticeType.equispaced, True
    )

    ref_vals = element.tabulate(0, eval_pts)[0]  # Shape: (n_pts, degree+1)
    vals = evaluate_cardinal_Bspline_basis(degree, eval_pts[:, 0])

    assert np.allclose(ref_vals, vals)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_spline_element_partition_of_unity(degree):
    """Test that B-spline basis functions sum to 1 (partition of unity)."""
    supdegree = degree + 5
    element = SplineElement(degree, supdegree)

    n_pts = 50
    eval_pts = basix.create_lattice(
        CellType.interval, n_pts, basix.LatticeType.equispaced, True
    )

    ref_vals = element.tabulate(0, eval_pts)[0]  # Shape: (n_pts, degree+1)
    sums = np.sum(ref_vals, axis=1)

    assert np.allclose(sums, 1.0)


def test_spline_element_non_negativity():
    """Test that B-spline basis functions are non-negative."""
    degree = 3
    supdegree = degree + 5
    element = SplineElement(degree, supdegree)

    n_pts = 100
    eval_pts = basix.create_lattice(
        CellType.interval, n_pts, basix.LatticeType.equispaced, True
    )

    ref_vals = element.tabulate(0, eval_pts)[0]  # Shape: (n_pts, degree+1)

    assert np.all(ref_vals >= -1e-14)  # Allow for small numerical errors
