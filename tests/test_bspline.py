"""Tests for B-spline functionality."""

import numpy as np

from dolfinx_iga.bspline import BSplineCurve, BSplineSurface


class TestBSplineCurve:
    """Test B-spline curve functionality."""

    def test_curve_creation(self, simple_control_points_2d):
        """Test basic curve creation."""
        degree = 2
        curve = BSplineCurve(simple_control_points_2d, degree)

        assert curve.degree == degree
        assert curve.n_control_points == len(simple_control_points_2d)
        assert len(curve.knot_vector) == len(simple_control_points_2d) + degree + 1

    def test_curve_evaluation_single_point(self, simple_control_points_2d):
        """Test curve evaluation at single parameter value."""
        degree = 2
        curve = BSplineCurve(simple_control_points_2d, degree)

        # Evaluate at parameter 0.5
        point = curve.evaluate(0.5)

        assert point.shape == (2,)  # 2D point
        assert isinstance(point, np.ndarray)

    def test_curve_evaluation_multiple_points(self, simple_control_points_2d):
        """Test curve evaluation at multiple parameter values."""
        degree = 2
        curve = BSplineCurve(simple_control_points_2d, degree)

        # Evaluate at multiple parameters
        u_vals = np.linspace(0, 1, 10)
        points = curve.evaluate(u_vals)

        assert points.shape == (10, 2)  # 10 points, 2D each

    def test_curve_endpoints(self, simple_control_points_2d):
        """Test that curve passes through first and last control points."""
        degree = 2
        curve = BSplineCurve(simple_control_points_2d, degree)

        # For clamped B-splines, curve should pass through endpoints
        start_point = curve.evaluate(0.0)
        end_point = curve.evaluate(1.0)

        np.testing.assert_allclose(start_point, simple_control_points_2d[0], atol=1e-10)
        np.testing.assert_allclose(end_point, simple_control_points_2d[-1], atol=1e-10)

    def test_curve_derivative(self, simple_control_points_2d):
        """Test curve derivative computation."""
        degree = 2
        curve = BSplineCurve(simple_control_points_2d, degree)

        # Evaluate derivative at parameter 0.5
        derivative = curve.derivative(0.5)

        assert derivative.shape == (2,)  # 2D derivative
        assert isinstance(derivative, np.ndarray)

    def test_custom_knot_vector(self, simple_control_points_2d):
        """Test curve with custom knot vector."""
        degree = 2
        knot_vector = np.array([0, 0, 0, 0.5, 1, 1, 1])

        curve = BSplineCurve(simple_control_points_2d, degree, knot_vector)

        np.testing.assert_array_equal(curve.knot_vector, knot_vector)

    def test_3d_curve(self, simple_control_points_3d):
        """Test 3D curve functionality."""
        degree = 2
        curve = BSplineCurve(simple_control_points_3d, degree)

        point = curve.evaluate(0.5)
        assert point.shape == (3,)  # 3D point


class TestBSplineSurface:
    """Test B-spline surface functionality."""

    def test_surface_creation(self, surface_control_points):
        """Test basic surface creation."""
        degree_u = 2
        degree_v = 2
        surface = BSplineSurface(surface_control_points, degree_u, degree_v)

        assert surface.degree_u == degree_u
        assert surface.degree_v == degree_v
        assert surface.n_u == surface_control_points.shape[0]
        assert surface.n_v == surface_control_points.shape[1]

    def test_surface_evaluation(self, surface_control_points):
        """Test surface evaluation."""
        degree_u = 2
        degree_v = 2
        surface = BSplineSurface(surface_control_points, degree_u, degree_v)

        # Single point evaluation
        point = surface.evaluate(0.5, 0.5)
        assert point.shape == (3,)  # 3D point

        # Multiple points evaluation
        u_vals = np.linspace(0, 1, 5)
        v_vals = np.linspace(0, 1, 5)
        points = surface.evaluate(u_vals, v_vals)
        assert points.shape == (5, 5, 3)  # 5x5 grid of 3D points

    def test_surface_corners(self, surface_control_points):
        """Test that surface passes through corner control points."""
        degree_u = 2
        degree_v = 2
        surface = BSplineSurface(surface_control_points, degree_u, degree_v)

        # Check corners
        corner_00 = surface.evaluate(0.0, 0.0)
        corner_10 = surface.evaluate(1.0, 0.0)
        corner_01 = surface.evaluate(0.0, 1.0)
        corner_11 = surface.evaluate(1.0, 1.0)

        np.testing.assert_allclose(corner_00, surface_control_points[0, 0], atol=1e-10)
        np.testing.assert_allclose(corner_10, surface_control_points[-1, 0], atol=1e-10)
        np.testing.assert_allclose(corner_01, surface_control_points[0, -1], atol=1e-10)
        np.testing.assert_allclose(
            corner_11, surface_control_points[-1, -1], atol=1e-10
        )
