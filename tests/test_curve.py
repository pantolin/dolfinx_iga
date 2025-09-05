"""Tests for curve module."""

import numpy as np
import pytest

from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.curve import BsplineCurve
from dolfinx_iga.splines.knots import (
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)


class TestBsplineCurveInit:
    """Test BsplineCurve initialization."""

    def test_valid_initialization(self):
        """Test valid BsplineCurve initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

        curve = BsplineCurve(space, control_points)

        assert curve.degree == 2
        assert curve.geom_dim == 2
        assert curve.rational is False
        assert curve.num_control_points == 3

    def test_rational_initialization(self):
        """Test rational BsplineCurve initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])

        curve = BsplineCurve(space, control_points, is_rational=True)

        assert curve.degree == 2
        assert curve.geom_dim == 2  # Excluding weight
        assert curve.rational is True
        assert curve.num_control_points == 3

    def test_integer_control_points_conversion(self):
        """Test that integer control points are converted to float64."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int32)

        curve = BsplineCurve(space, control_points)

        assert curve.dtype == np.float64
        np.testing.assert_array_equal(
            curve.control_points,
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64),
        )

    def test_control_points_dimension_mismatch_error(self):
        """Test that control points dimension mismatch raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0]])  # Wrong number of points

        with pytest.raises(ValueError, match="Number of control points must match"):
            BsplineCurve(space, control_points)

    def test_invalid_control_points_dimension_error(self):
        """Test that invalid control points dimension raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[[0.0, 0.0]]])  # 3D array

        with pytest.raises(ValueError, match="Control points must be a 2D array"):
            BsplineCurve(space, control_points)

    def test_rational_insufficient_coordinates_error(self):
        """Test that rational curves with insufficient coordinates raise ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0], [1.0], [2.0]])  # Only 1 coordinate

        with pytest.raises(
            ValueError, match="Invalid number of coordinates for rational curves"
        ):
            BsplineCurve(space, control_points, is_rational=True)

    def test_dtype_mismatch_error(self):
        """Test that dtype mismatch between space and control points raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64
        )

        with pytest.raises(
            ValueError, match="Control points and space must have the same dtype"
        ):
            BsplineCurve(space, control_points)


class TestBsplineCurveProperties:
    """Test BsplineCurve properties."""

    def test_space_property(self):
        """Test space property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.space is space

    def test_control_points_property(self):
        """Test control_points property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        np.testing.assert_array_equal(curve.control_points, control_points)

    def test_rational_property(self):
        """Test rational property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
        curve = BsplineCurve(space, control_points, is_rational=True)

        assert curve.rational is True

    def test_dimension_property(self):
        """Test dimension property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)

        # Test 2D curve
        control_points_2d = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve_2d = BsplineCurve(space, control_points_2d)
        assert curve_2d.geom_dim == 2

        # Test 3D curve
        control_points_3d = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]
        )
        curve_3d = BsplineCurve(space, control_points_3d)
        assert curve_3d.geom_dim == 3

        # Test rational 2D curve
        control_points_rational = np.array(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]]
        )
        curve_rational = BsplineCurve(space, control_points_rational, is_rational=True)
        assert curve_rational.geom_dim == 2  # Excluding weight

    def test_dtype_property(self):
        """Test dtype property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.dtype == np.float64

    def test_num_control_points_property(self):
        """Test num_control_points property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.num_control_points == 3

    def test_periodic_property(self):
        """Test periodic property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.periodic is False

    def test_degree_property(self):
        """Test degree property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.degree == 2


class TestBsplineCurveMethods:
    """Test BsplineCurve methods."""

    def test_get_domain(self):
        """Test get_domain method."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        domain = curve.get_domain()
        assert domain == (0.0, 1.0)

    def test_evaluate_single_point(self):
        """Test evaluation at single point."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        result = curve.evaluate(0.5)

        # Should be 2D point (single point returns 1D array)
        assert result.shape == (2,)
        # Should be within reasonable bounds
        assert np.all(np.isfinite(result))

    def test_evaluate_multiple_points(self):
        """Test evaluation at multiple points."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        pts = [0.0, 0.5, 1.0]
        result = curve.evaluate(pts)

        # Should be (3, 2) array
        assert result.shape == (3, 2)
        # All points should be finite
        assert np.all(np.isfinite(result))

    def test_evaluate_rational_curve(self):
        """Test evaluation of rational curve."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
        curve = BsplineCurve(space, control_points, is_rational=True)

        result = curve.evaluate(0.5)

        # Should be 2D point (excluding weight)
        assert result.shape == (2,)
        # Should be finite
        assert np.all(np.isfinite(result))

    def test_evaluate_at_boundaries(self):
        """Test evaluation at domain boundaries."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        # Evaluate at boundaries
        result_start = curve.evaluate(0.0)
        result_end = curve.evaluate(1.0)

        # Should be finite
        assert np.all(np.isfinite(result_start))
        assert np.all(np.isfinite(result_end))

    def test_evaluate_periodic_curve(self):
        """Test evaluation of periodic curve."""
        knots = create_uniform_periodic_knot_vector(3, 2, start=0.0, end=1.0)
        space = Bspline1D(knots, 2, periodic=True)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        result = curve.evaluate(0.5)

        # Should be finite
        assert np.all(np.isfinite(result))

    def test_evaluate_outside_domain_error(self):
        """Test that evaluation outside domain raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space = Bspline1D(knots, degree)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        with pytest.raises(ValueError, match="outside the knot vector domain"):
            curve.evaluate(-0.1)


class TestBsplineCurveWithDifferentSpaces:
    """Test BsplineCurve with different B-spline spaces."""

    def test_with_uniform_open_knot_vector(self):
        """Test curve with uniform open knot vector."""
        knots = create_uniform_open_knot_vector(3, 2, start=0.0, end=1.0)
        space = Bspline1D(knots, 2)
        # Need correct number of control points for the space
        num_basis = space.get_num_basis()
        control_points = np.array(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [1.5, 1.0], [2.0, 0.0]]
        )[:num_basis]
        curve = BsplineCurve(space, control_points)

        assert curve.degree == 2
        assert curve.geom_dim == 2
        assert curve.num_control_points == num_basis
        assert curve.periodic is False
        assert curve.get_domain() == (0.0, 1.0)

    def test_with_periodic_knot_vector(self):
        """Test curve with periodic knot vector."""
        knots = create_uniform_periodic_knot_vector(3, 2, start=0.0, end=1.0)
        space = Bspline1D(knots, 2, periodic=True)
        control_points = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.degree == 2
        assert curve.geom_dim == 2
        assert curve.periodic is True
        assert curve.get_domain() == (0.0, 1.0)

    def test_with_high_degree(self):
        """Test curve with high degree."""
        knots = [0.0] * 6 + [1.0] * 6  # Degree 5
        space = Bspline1D(knots, 5)
        # Need correct number of control points for the space
        num_basis = space.get_num_basis()
        # Create enough control points for the space
        control_points = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0], [5.0, 1.0]]
        )[:num_basis]
        curve = BsplineCurve(space, control_points)

        assert curve.degree == 5
        assert curve.num_control_points == num_basis

    def test_with_3d_control_points(self):
        """Test curve with 3D control points."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space = Bspline1D(knots, 2)
        control_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.geom_dim == 3
        result = curve.evaluate(0.5)
        assert result.shape == (3,)


class TestBsplineCurveEdgeCases:
    """Test BsplineCurve edge cases."""

    def test_degree_zero_curve(self):
        """Test curve with degree 0."""
        knots = [0.0, 1.0]
        space = Bspline1D(knots, 0)
        control_points = np.array([[1.0, 2.0]])
        curve = BsplineCurve(space, control_points)

        assert curve.degree == 0
        assert curve.num_control_points == 1

        # Should evaluate to the control point
        result = curve.evaluate(0.5)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_single_control_point(self):
        """Test curve with single control point."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space = Bspline1D(knots, 2)
        # Need correct number of control points for the space
        num_basis = space.get_num_basis()
        control_points = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])[:num_basis]
        curve = BsplineCurve(space, control_points)

        assert curve.num_control_points == num_basis

        # Should evaluate to a point on the curve
        result = curve.evaluate(0.5)
        # The result should be finite and have correct shape
        assert np.all(np.isfinite(result))
        assert result.shape == (2,)

    def test_float32_precision(self):
        """Test curve with float32 precision."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        space = Bspline1D(knots, 2)
        control_points = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32
        )
        curve = BsplineCurve(space, control_points)

        assert curve.dtype == np.float32
        result = curve.evaluate(0.5)
        assert result.dtype == np.float32


class TestBsplineCurveIntegration:
    """Integration tests for BsplineCurve."""

    def test_consistency_across_evaluations(self):
        """Test consistency across multiple evaluations."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space = Bspline1D(knots, 2)
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        curve = BsplineCurve(space, control_points)

        # Evaluate at multiple points
        pts = np.linspace(0.0, 1.0, 11)
        results = curve.evaluate(pts)

        # All results should be finite
        assert np.all(np.isfinite(results))

        # Results should have correct shape
        assert results.shape == (11, 2)

    def test_rational_vs_polynomial_consistency(self):
        """Test consistency between rational and polynomial curves."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space = Bspline1D(knots, 2)

        # Polynomial curve
        control_points_poly = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        curve_poly = BsplineCurve(space, control_points_poly, is_rational=False)

        # Rational curve with unit weights
        control_points_rational = np.array(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]]
        )
        curve_rational = BsplineCurve(space, control_points_rational, is_rational=True)

        # Should give same results
        pts = [0.0, 0.5, 1.0]
        result_poly = curve_poly.evaluate(pts)
        result_rational = curve_rational.evaluate(pts)

        np.testing.assert_array_almost_equal(result_poly, result_rational)

    def test_periodic_vs_non_periodic_consistency(self):
        """Test consistency between periodic and non-periodic curves."""
        # Create equivalent spaces
        knots_open = create_uniform_open_knot_vector(3, 2, start=0.0, end=1.0)
        knots_periodic = create_uniform_periodic_knot_vector(3, 2, start=0.0, end=1.0)

        space_open = Bspline1D(knots_open, 2, periodic=False)
        space_periodic = Bspline1D(knots_periodic, 2, periodic=True)

        # Create curves with same control points (adjusted for different basis counts)
        num_basis_open = space_open.get_num_basis()
        num_basis_periodic = space_periodic.get_num_basis()
        control_points_open = np.array(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [1.5, 1.0], [2.0, 0.0]]
        )[:num_basis_open]
        control_points_periodic = np.array(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [1.5, 1.0]]
        )[:num_basis_periodic]

        curve_open = BsplineCurve(space_open, control_points_open)
        curve_periodic = BsplineCurve(space_periodic, control_points_periodic)

        # Both should have same domain
        assert curve_open.get_domain() == curve_periodic.get_domain()

        # Both should evaluate without errors
        pts = [0.0, 0.5, 1.0]
        result_open = curve_open.evaluate(pts)
        result_periodic = curve_periodic.evaluate(pts)

        assert np.all(np.isfinite(result_open))
        assert np.all(np.isfinite(result_periodic))


class TestBsplineCurveEvaluation:
    """Test BsplineCurve evaluation."""

    def test_circle(self):
        """Test evaluation of a circle."""
        knots = create_uniform_open_knot_vector(4, 2, 0, start=0.0, end=1.0)
        center = np.array([1.0, 2.0])
        radius = 3.5

        c = np.sqrt(2.0) / 2.0
        x0 = center[0] - radius
        x1 = center[0]
        x2 = center[0] + radius
        y0 = center[1] - radius
        y1 = center[1]
        y2 = center[1] + radius

        control_points = np.array(
            [
                [x2, y1, 1.0],
                [x2 * c, y2 * c, c],
                [x1, y2, 1.0],
                [x0 * c, y2 * c, c],
                [x0, y1, 1.0],
                [x0 * c, y0 * c, c],
                [x1, y0, 1.0],
                [x2 * c, y0 * c, c],
                [x2, y1, 1.0],
            ]
        )

        space = Bspline1D(knots, 2)
        curve = BsplineCurve(space, control_points, is_rational=True)

        n_pts = 10
        pts = np.linspace(0.0, 1.0, n_pts)
        result = curve.evaluate(pts)

        assert result.shape == (n_pts, 2)
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))

        # Check that the result is close to the circle
        distance = np.linalg.norm(result - center, axis=1)
        np.testing.assert_array_almost_equal(distance, radius)
