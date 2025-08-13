"""Tests for NURBS functionality."""

import numpy as np

from dolfinx_iga.nurbs import NURBSCurve, NURBSSurface


class TestNURBSCurve:
    """Test NURBS curve functionality."""

    def test_nurbs_creation(self, simple_control_points_2d, simple_weights):
        """Test basic NURBS curve creation."""
        degree = 2
        curve = NURBSCurve(simple_control_points_2d, simple_weights, degree)

        assert curve.degree == degree
        assert len(curve.weights) == len(simple_control_points_2d)

    def test_nurbs_evaluation(self, simple_control_points_2d, simple_weights):
        """Test NURBS curve evaluation."""
        degree = 2
        curve = NURBSCurve(simple_control_points_2d, simple_weights, degree)

        # Single point evaluation
        point = curve.evaluate(0.5)
        assert point.shape == (2,)  # 2D point

        # Multiple points evaluation
        u_vals = np.linspace(0, 1, 10)
        points = curve.evaluate(u_vals)
        assert points.shape == (10, 2)

    def test_nurbs_vs_bspline_unit_weights(self, simple_control_points_2d):
        """Test that NURBS with unit weights equals B-spline."""
        from dolfinx_iga.bspline import BSplineCurve

        degree = 2
        unit_weights = np.ones(len(simple_control_points_2d))

        nurbs_curve = NURBSCurve(simple_control_points_2d, unit_weights, degree)
        bspline_curve = BSplineCurve(simple_control_points_2d, degree)

        # Evaluate at same points
        u_vals = np.linspace(0, 1, 10)
        nurbs_points = nurbs_curve.evaluate(u_vals)
        bspline_points = bspline_curve.evaluate(u_vals)

        np.testing.assert_allclose(nurbs_points, bspline_points, atol=1e-10)

    def test_nurbs_derivative(self, simple_control_points_2d, simple_weights):
        """Test NURBS curve derivative."""
        degree = 2
        curve = NURBSCurve(simple_control_points_2d, simple_weights, degree)

        derivative = curve.derivative(0.5)
        assert derivative.shape == (2,)

    def test_rational_circle_arc(self):
        """Test NURBS representation of a circular arc."""
        # Quarter circle arc from (1,0) to (0,1)
        control_points = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        weights = np.array([1.0, 1.0 / np.sqrt(2), 1.0])
        degree = 2

        curve = NURBSCurve(control_points, weights, degree)

        # Check endpoints
        start = curve.evaluate(0.0)
        end = curve.evaluate(1.0)

        np.testing.assert_allclose(start, [1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(end, [0.0, 1.0], atol=1e-10)

        # Check midpoint (should be on unit circle)
        mid = curve.evaluate(0.5)
        radius = np.linalg.norm(mid)
        np.testing.assert_allclose(radius, 1.0, atol=1e-10)


class TestNURBSSurface:
    """Test NURBS surface functionality."""

    def test_nurbs_surface_creation(self, surface_control_points, surface_weights):
        """Test basic NURBS surface creation."""
        degree_u = 2
        degree_v = 2
        surface = NURBSSurface(
            surface_control_points, surface_weights, degree_u, degree_v
        )

        assert surface.degree_u == degree_u
        assert surface.degree_v == degree_v
        assert surface.weights.shape == surface_control_points.shape[:2]

    def test_nurbs_surface_evaluation(self, surface_control_points, surface_weights):
        """Test NURBS surface evaluation."""
        degree_u = 2
        degree_v = 2
        surface = NURBSSurface(
            surface_control_points, surface_weights, degree_u, degree_v
        )

        # Single point evaluation
        point = surface.evaluate(0.5, 0.5)
        assert point.shape == (3,)  # 3D point

        # Multiple points evaluation
        u_vals = np.linspace(0, 1, 3)
        v_vals = np.linspace(0, 1, 3)
        points = surface.evaluate(u_vals, v_vals)
        assert points.shape == (3, 3, 3)  # 3x3 grid of 3D points

    def test_nurbs_vs_bspline_surface_unit_weights(self, surface_control_points):
        """Test that NURBS surface with unit weights equals B-spline surface."""
        from dolfinx_iga.bspline import BSplineSurface

        degree_u = 2
        degree_v = 2
        unit_weights = np.ones(surface_control_points.shape[:2])

        nurbs_surface = NURBSSurface(
            surface_control_points, unit_weights, degree_u, degree_v
        )
        bspline_surface = BSplineSurface(surface_control_points, degree_u, degree_v)

        # Evaluate at same points
        u_vals = np.linspace(0, 1, 3)
        v_vals = np.linspace(0, 1, 3)
        nurbs_points = nurbs_surface.evaluate(u_vals, v_vals)
        bspline_points = bspline_surface.evaluate(u_vals, v_vals)

        np.testing.assert_allclose(nurbs_points, bspline_points, atol=1e-10)
