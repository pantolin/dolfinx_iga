import basix
import numpy as np
import pytest
from basix import CellType

from dolfinx_iga.splines.basis import evaluate_cardinal_Bspline_basis
from dolfinx_iga.splines.element import create_cardinal_Bspline_element


@pytest.mark.parametrize(
    "degrees",
    [
        # 1D
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        # 2D uniform
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        # 2D mixed
        (1, 2),
        (2, 3),
        (1, 4),
        (3, 5),
        # 3D uniform
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        # 3D mixed
        (1, 2, 1),
        (2, 3, 2),
        (1, 2, 3),
        (3, 4, 2),
    ],
)
def test_spline_element_tabulate_vs_cardinal_basis(degrees):
    """Test that SplineElement.tabulate matches evaluate_cardinal_Bspline_basis."""

    element = create_cardinal_Bspline_element(list(degrees))

    n_pts_dir = 20

    dim = len(degrees)
    cell_type = [CellType.interval, CellType.quadrilateral, CellType.hexahedron][
        dim - 1
    ]

    pts = basix.create_lattice(
        cell_type, n_pts_dir - 1, basix.LatticeType.equispaced, True
    )

    vals = element.tabulate(0, pts)[0]
    ref_vals = evaluate_cardinal_Bspline_basis(degrees, pts)
    assert np.allclose(ref_vals, vals)
