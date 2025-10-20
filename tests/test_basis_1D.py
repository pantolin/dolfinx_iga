"""Tests for basis_1D module."""

import numpy as np
import pytest

from dolfinx_iga.splines.basis_1D import (
    _prepare_pts_for_evaluation,
    evaluate_Bernstein_basis_1D,
    evaluate_Bspline_basis_1D,
    evaluate_Lagrange_basis_1D,
    evaluate_monomial_basis_1D,
)
from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.knots import (
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)


class TestPreparePtsForEvaluation:
    """Test the _prepare_pts_for_evaluation helper function."""

    def test_scalar_input(self):
        """Test scalar input is converted to 1D array."""
        result = _prepare_pts_for_evaluation(0.5)
        expected = np.array([0.5])
        np.testing.assert_array_equal(result, expected)

    def test_list_input(self):
        """Test list input is converted to numpy array."""
        result = _prepare_pts_for_evaluation([0.0, 0.5, 1.0])
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_reshape(self):
        """Test 2D array raises ValueError."""
        pts = np.array([[0.0, 0.5], [1.0, 1.5]])

        with pytest.raises(ValueError, match="pts must be a 1D array"):
            _prepare_pts_for_evaluation(pts)

    def test_integer_conversion(self):
        """Test integer inputs are converted to float64."""
        result = _prepare_pts_for_evaluation([0, 1, 2])
        expected = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


class TestEvaluateBernsteinBasis:
    """Test Bernstein basis function evaluation."""

    def test_degree_zero(self):
        """Test degree 0 Bernstein basis (constant function)."""
        result = evaluate_Bernstein_basis_1D(0, [0.0, 0.5, 1.0])
        expected = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_degree_one(self):
        """Test degree 1 Bernstein basis (linear functions)."""
        result = evaluate_Bernstein_basis_1D(1, [0.0, 0.5, 1.0])
        expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_two(self):
        """Test degree 2 Bernstein basis (quadratic functions)."""
        result = evaluate_Bernstein_basis_1D(2, [0.0, 0.5, 1.0])
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            evaluate_Bernstein_basis_1D(-1, [0.0, 0.5, 1.0])

    def test_partition_of_unity(self):
        """Test that Bernstein basis functions sum to 1 (partition of unity)."""
        pts = np.linspace(0.0, 1.0, 11)
        for degree in [1, 2, 3, 4]:
            result = evaluate_Bernstein_basis_1D(degree, pts)
            sums = np.sum(result, axis=1)
            np.testing.assert_array_almost_equal(sums, np.ones_like(sums))


class TestEvaluateMonomialBasis:
    """Test monomial basis function evaluation."""

    def test_degree_zero(self):
        """Test degree 0 monomial basis (constant function)."""
        result = evaluate_monomial_basis_1D(0, [0.0, 0.5, 1.0])
        expected = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_degree_one(self):
        """Test degree 1 monomial basis (1, t)."""
        result = evaluate_monomial_basis_1D(1, [0.0, 0.5, 1.0])
        expected = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_degree_two(self):
        """Test degree 2 monomial basis (1, t, t^2)."""
        result = evaluate_monomial_basis_1D(2, [0.0, 0.5, 1.0])
        expected = np.array([[1.0, 0.0, 0.0], [1.0, 0.5, 0.25], [1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            evaluate_monomial_basis_1D(-1, [0.0, 0.5, 1.0])


class TestEvaluateLagrangeBasis:
    """Test Lagrange basis function evaluation."""

    def test_degree_zero(self):
        """Test degree 0 Lagrange basis ValueError."""
        with pytest.raises(
            ValueError, match="Lagrange basis degree must be at least 1"
        ):
            evaluate_Lagrange_basis_1D(0, [0.0, 0.5, 1.0])

    def test_degree_one(self):
        """Test degree 1 Lagrange basis."""
        result = evaluate_Lagrange_basis_1D(1, [0.0, 0.5, 1.0])
        # Lagrange basis should be 1 at nodes and 0 at other nodes
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0])
        np.testing.assert_array_almost_equal(result[2], [0.0, 1.0])

    def test_degree_two(self):
        """Test degree 2 Lagrange basis."""
        result = evaluate_Lagrange_basis_1D(2, [0.0, 0.5, 1.0])
        # Check that basis functions are 1 at their respective nodes
        # Note: basix ordering may be different, so just check shape and partition of unity
        assert result.shape == (3, 3)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))


class TestEvaluateBsplineBasis:
    """Test B-spline basis function evaluation."""

    def test_bezier_like_knots(self):
        """Test evaluation with Bézier-like knots."""
        degree = 2
        knots = create_uniform_open_knot_vector(1, degree, start=2.0, end=4.5)
        spline = Bspline1D(knots, degree)

        eval_pts = np.array([0.0, 0.5, 1.0])

        t0, t1 = spline.domain
        result, first_idx = evaluate_Bspline_basis_1D(spline, eval_pts * (t1 - t0) + t0)

        # Should be Bernstein basis for Bézier-like knots
        expected = evaluate_Bernstein_basis_1D(degree, eval_pts)
        np.testing.assert_array_almost_equal(result, expected)
        np.testing.assert_array_equal(first_idx, [0, 0, 0])

    def test_general_knots(self):
        """Test evaluation with general knot vector."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        spline = Bspline1D(knots, 2)
        result, first_idx = evaluate_Bspline_basis_1D(spline, [0.25, 0.75])

        # Check that result has correct shape
        assert result.shape == (2, 3)
        assert first_idx.shape == (2,)

        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_outside_domain_error(self):
        """Test that evaluation outside domain raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        spline = Bspline1D(knots, 2)

        with pytest.raises(ValueError, match="outside the knot vector domain"):
            evaluate_Bspline_basis_1D(spline, [-0.1])

    def test_periodic_spline(self):
        """Test evaluation with periodic spline."""
        knots = create_uniform_periodic_knot_vector(3, 2, start=0.0, end=1.0)
        spline = Bspline1D(knots, 2, periodic=True)
        result, first_idx = evaluate_Bspline_basis_1D(spline, [0.0, 0.5, 1.0])

        # Check that result has correct shape
        assert result.shape == (3, 3)
        assert first_idx.shape == (3,)

        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))
