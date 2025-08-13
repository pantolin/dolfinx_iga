"""Test package imports and basic functionality."""

import numpy as np


def test_package_import():
    """Test that the package can be imported."""
    import dolfinx_iga

    assert hasattr(dolfinx_iga, "__version__")


def test_bspline_import():
    """Test B-spline imports."""
    from dolfinx_iga import BSplineCurve, BSplineSurface

    assert BSplineCurve is not None
    assert BSplineSurface is not None


def test_nurbs_import():
    """Test NURBS imports."""
    from dolfinx_iga import NURBSCurve, NURBSSurface

    assert NURBSCurve is not None
    assert NURBSSurface is not None


def test_utils_import():
    """Test utils imports."""
    from dolfinx_iga import basis_functions, knot_vector_utils

    assert knot_vector_utils is not None
    assert basis_functions is not None


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    from dolfinx_iga import BSplineCurve

    # Simple 2D control points
    control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    degree = 2

    # Create curve
    curve = BSplineCurve(control_points, degree)

    # Evaluate at midpoint
    point = curve.evaluate(0.5)

    # Basic checks
    assert isinstance(point, np.ndarray)
    assert point.shape == (2,)
    assert np.isfinite(point).all()
