"""Tests for change_basis_1D module."""

import numpy as np
import pytest
from basix import LagrangeVariant

from dolfinx_iga.splines.basis_1D import (
    evaluate_Bernstein_basis_1D,
    evaluate_Bspline_basis_1D,
    evaluate_cardinal_Bspline_basis_1D,
    evaluate_Lagrange_basis_1D,
)
from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.change_basis_1D import (
    _compute_gauss_quadrature,
    _create_change_basis_operator,
    create_Bernstein_to_cardinal_basis_operator,
    create_Bernstein_to_Lagrange_basis_operator,
    create_Bezier_extraction_operators,
    create_cardinal_extraction_operators,
    create_cardinal_to_Bernstein_basis_operator,
    create_Lagrange_extraction_operators,
    create_Lagrange_to_Bernstein_basis_operator,
)
from dolfinx_iga.splines.knots import create_uniform_open_knot_vector
from dolfinx_iga.utils.tolerance import get_default_tolerance
from dolfinx_iga.utils.types import FloatArray_32_64


class TestComputeGaussQuadrature:
    """Test the _compute_gauss_quadrature function."""

    def test_single_point(self):
        """Test quadrature with single point."""
        points, weights = _compute_gauss_quadrature(1)

        assert len(points) == 1
        assert len(weights) == 1
        np.testing.assert_allclose(points[0], 0.5)  # Should be at midpoint
        np.testing.assert_allclose(
            weights[0], 1.0
        )  # Weight should be 1.0 for [0,1] interval

    def test_multiple_points(self):
        """Test quadrature with multiple points."""
        points, weights = _compute_gauss_quadrature(3)

        assert len(points) == 3
        assert len(weights) == 3

        # Check that points are in [0, 1]
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)

        # Check that weights sum to 1 (for [0, 1] interval)
        np.testing.assert_allclose(np.sum(weights), 1.0)

    def test_non_positive_points_error(self):
        """Test that non positive number of points raises AssertionError."""
        with pytest.raises(AssertionError, match="Number of points must be positive"):
            _compute_gauss_quadrature(0)
        with pytest.raises(AssertionError, match="Number of points must be positive"):
            _compute_gauss_quadrature(-1)

    def test_high_order_quadrature(self):
        """Test high-order quadrature."""
        points, weights = _compute_gauss_quadrature(10)

        assert len(points) == 10
        assert len(weights) == 10

        # Check that points are in [0, 1]
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)

        # Check that weights sum to 1
        np.testing.assert_allclose(np.sum(weights), 1.0)


class TestCreateChangeBasisOperator:
    """Test the _create_change_basis_operator function."""

    def test_identity_transformation(self):
        """Test transformation between identical bases."""

        def basis_func(pts):
            return np.ones((len(pts), 1))

        result = _create_change_basis_operator(
            basis_func, basis_func, n_quad_pts=3, dtype=np.float64
        )

        # Should be identity matrix
        np.testing.assert_array_almost_equal(result, np.eye(1))

    def test_non_positive_quadrature_points_error(self):
        """Test that non positive number of quadrature points raise ValueError."""

        def basis_func(pts):
            return np.ones((len(pts), 1))

        with pytest.raises(
            ValueError, match="Number of quadrature points must be positive"
        ):
            _create_change_basis_operator(
                basis_func, basis_func, n_quad_pts=0, dtype=np.float64
            )

        with pytest.raises(
            ValueError, match="Number of quadrature points must be positive"
        ):
            _create_change_basis_operator(
                basis_func, basis_func, n_quad_pts=-1, dtype=np.float64
            )

    def test_different_dtypes(self):
        """Test with different data types."""

        def basis_func(pts):
            return np.ones((len(pts), 1))

        result_f64 = _create_change_basis_operator(
            basis_func, basis_func, n_quad_pts=3, dtype=np.float64
        )
        result_f32 = _create_change_basis_operator(
            basis_func, basis_func, n_quad_pts=3, dtype=np.float32
        )

        assert result_f64.dtype == np.float64
        # Note: The function might return float64 due to internal computations
        assert result_f32.dtype in (np.float32, np.float64)


class TestLagrangeToBernsteinBasisOperator:
    """Test the create_Lagrange_to_Bernstein_basis_operator function."""

    def test_degree_zero_error(self):
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Lagrange_to_Bernstein_basis_operator(0)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Lagrange_to_Bernstein_basis_operator(-1)

    def test_degree_one(self):
        """Test degree 1 transformation."""
        result = create_Lagrange_to_Bernstein_basis_operator(1)

        assert result.shape == (2, 2)
        # Should be invertible
        assert np.linalg.det(result) != 0

    def test_degree_two(self):
        """Test degree 2 transformation."""
        result = create_Lagrange_to_Bernstein_basis_operator(2)

        assert result.shape == (3, 3)
        # Should be invertible
        assert np.linalg.det(result) != 0

    def test_different_variants(self):
        """Test with different Lagrange variants."""
        result_equispaced = create_Lagrange_to_Bernstein_basis_operator(
            2, LagrangeVariant.equispaced
        )
        result_gll = create_Lagrange_to_Bernstein_basis_operator(
            2, LagrangeVariant.gll_warped
        )

        # Should have same shape but may have same values for this case
        assert result_equispaced.shape == result_gll.shape
        # Note: For degree 2, these variants might produce the same result

    def test_values(self):
        """Test that Lagrange evaluations transformed with the operator return Bernstein evaluations."""
        for degree in [1, 2, 3, 4]:
            C = create_Lagrange_to_Bernstein_basis_operator(degree)
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = evaluate_Bernstein_basis_1D(degree, tt)
            lagranges = evaluate_Lagrange_basis_1D(degree, tt)
            np.testing.assert_array_almost_equal(bernsteins, lagranges @ C.T)


class TestBernsteinToLagrangeBasisOperator:
    """Test the create_Bernstein_to_Lagrange_basis_operator function."""

    def test_degree_zero_error(self):
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Bernstein_to_Lagrange_basis_operator(0)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Bernstein_to_Lagrange_basis_operator(-1)

    def test_inverse_relationship(self):
        """Test that Bernstein to Lagrange is inverse of Lagrange to Bernstein."""
        degree = 2
        lagrange_to_bernstein = create_Lagrange_to_Bernstein_basis_operator(degree)
        bernstein_to_lagrange = create_Bernstein_to_Lagrange_basis_operator(degree)

        # Should be inverse matrices
        identity = lagrange_to_bernstein @ bernstein_to_lagrange
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    def test_values(self):
        """Test that Bernstein evaluations transformed with the operator return Lagrange evaluations."""
        for degree in [1, 2, 3, 4]:
            C = create_Bernstein_to_Lagrange_basis_operator(degree)
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = evaluate_Bernstein_basis_1D(degree, tt)
            lagranges = evaluate_Lagrange_basis_1D(degree, tt)
            np.testing.assert_array_almost_equal(bernsteins @ C.T, lagranges)


class TestBernsteinToCardinalBasisOperator:
    """Test the create_Bernstein_to_cardinal_basis_operator function."""

    def test_degree_zero(self):
        """Test degree 0 transformation."""
        result = create_Bernstein_to_cardinal_basis_operator(0)

        assert result.shape == (1, 1)
        np.testing.assert_array_almost_equal(result, np.eye(1))

    def test_degree_one(self):
        """Test degree 1 transformation."""
        result = create_Bernstein_to_cardinal_basis_operator(1)

        assert result.shape == (2, 2)
        # Should be invertible
        assert np.linalg.det(result) != 0

    def test_degree_two(self):
        """Test degree 2 transformation."""
        result = create_Bernstein_to_cardinal_basis_operator(2)

        assert result.shape == (3, 3)
        # Should be invertible
        assert np.linalg.det(result) != 0

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            create_Bernstein_to_cardinal_basis_operator(-1)

    def test_values(self):
        """Test that Bernstein evaluations transformed with the operator return cardinal evaluations."""
        for degree in [1, 2, 3, 4]:
            C = create_Bernstein_to_cardinal_basis_operator(degree)
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = evaluate_Bernstein_basis_1D(degree, tt)
            cardinals = evaluate_cardinal_Bspline_basis_1D(degree, tt)
            np.testing.assert_array_almost_equal(bernsteins @ C.T, cardinals)


class TestCardinalToBernsteinBasisOperator:
    """Test the create_cardinal_to_Bernstein_basis_operator function."""

    def test_inverse_relationship(self):
        """Test that cardinal to Bernstein is inverse of Bernstein to cardinal."""
        degree = 2
        bernstein_to_cardinal = create_Bernstein_to_cardinal_basis_operator(degree)
        cardinal_to_bernstein = create_cardinal_to_Bernstein_basis_operator(degree)

        # Should be inverse matrices
        identity = bernstein_to_cardinal @ cardinal_to_bernstein
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            create_cardinal_to_Bernstein_basis_operator(-1)

    def test_values(self):
        """Test that cardinal evaluations transformed with the operator return Bernstein evaluations."""
        for degree in [1, 2, 3, 4]:
            C = create_cardinal_to_Bernstein_basis_operator(degree)
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = evaluate_Bernstein_basis_1D(degree, tt)
            cardinals = evaluate_cardinal_Bspline_basis_1D(degree, tt)
            np.testing.assert_array_almost_equal(bernsteins, cardinals @ C.T)


def create_uniform_spline(
    num_intervals: int, degree: int, continuity: int = None
) -> Bspline1D:
    """Create a uniform spline with the given degree for testing purposes."""
    knots = create_uniform_open_knot_vector(num_intervals, degree, continuity)
    return Bspline1D(knots, degree)


def create_non_open_spline_1() -> Bspline1D:
    """Create a non-open spline of degree 4, and reduced continuity, for testing purposes."""
    degree = 4
    knots = np.array(
        [0, 1, 6, 6, 10, 11, 12, 13, 13, 13, 13],
        dtype=np.float64,
    )
    return Bspline1D(knots, degree)


def create_non_open_spline_2() -> Bspline1D:
    """Create a non-open spline of degree 4, and maximum continuity, for testing purposes."""
    degree = 4
    knots = np.array(
        [-1, 0, 1, 3, 6, 10, 11, 12, 13, 13, 13, 13, 13],
        dtype=np.float64,
    )
    return Bspline1D(knots, degree)


def extraction_operators_tester(
    spline: Bspline1D, ref_eval: callable, ops: FloatArray_32_64, n_sample_pts: int = 10
):
    """Test the extraction operators."""

    tol = get_default_tolerance(spline.dtype)
    # We prevent here to evaluate at 1.0, as it would evaluate at the next
    # interval of the spline.
    tt = np.linspace(0.0, 1.0 - tol, n_sample_pts)
    ref_vals = ref_eval(spline.degree, tt)

    n_intervals = spline.num_intervals
    unique_knots = spline.get_unique_knots_and_multiplicity(in_domain=True)[0]

    for i in range(n_intervals):
        new_tt = tt * (unique_knots[i + 1] - unique_knots[i]) + unique_knots[i]
        bspline_vals, _ = evaluate_Bspline_basis_1D(spline, new_tt)
        C = ops[i]
        np.testing.assert_array_almost_equal(bspline_vals, ref_vals @ C.T)


class TestBezierExtractionOperators:
    """Test the create_Bezier_extraction_operators function."""

    def test_bezier_like_spline(self):
        """Test extraction operators for Bézier-like spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        spline = Bspline1D(knots, 2)
        result = create_Bezier_extraction_operators(spline)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

        # For Bézier-like knots, extraction matrix should be identity
        np.testing.assert_array_almost_equal(result[0], np.eye(3))

    def test_general_spline(self):
        """Test extraction operators for general spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        spline = Bspline1D(knots, 2)
        result = create_Bezier_extraction_operators(spline)

        # Should have 2 intervals, 3x3 extraction matrices
        assert result.shape == (2, 3, 3)

        # Check that matrices are not identity (since not Bézier-like)
        assert not np.allclose(result[0], np.eye(3))
        assert not np.allclose(result[1], np.eye(3))

    def test_values(self):
        """Test the extraction operators."""
        n_intervals = 10
        splines = [
            create_uniform_spline(n_intervals, degree) for degree in [1, 2, 3, 4]
        ]
        splines.append(create_non_open_spline_1())
        splines.append(create_non_open_spline_2())

        for spline in splines:
            ops = create_Bezier_extraction_operators(spline)
            extraction_operators_tester(spline, evaluate_Bernstein_basis_1D, ops)


class TestLagrangeExtractionOperators:
    """Test the create_Lagrange_extraction_operators function."""

    def test_degree_zero_error(self):
        """Test that degree lower than 1 raises ValueError."""
        knots = [0.0, 1.0]
        spline = Bspline1D(knots, 0)
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Lagrange_extraction_operators(spline)

    def test_bezier_like_spline(self):
        """Test Lagrange extraction operators for Bézier-like spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        spline = Bspline1D(knots, 2)
        result = create_Lagrange_extraction_operators(spline)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

    def test_different_variants(self):
        """Test with different Lagrange variants."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        spline = Bspline1D(knots, 2)

        result_equispaced = create_Lagrange_extraction_operators(
            spline, LagrangeVariant.equispaced
        )
        result_gll = create_Lagrange_extraction_operators(
            spline, LagrangeVariant.gll_warped
        )

        # Should have same shape but may have same values for this case
        assert result_equispaced.shape == result_gll.shape
        # Note: For degree 2, these variants might produce the same result

    def test_values(self):
        """Test the extraction operators."""
        n_intervals = 10
        splines = [
            create_uniform_spline(n_intervals, degree) for degree in [1, 2, 3, 4]
        ]
        splines.append(create_non_open_spline_1())
        splines.append(create_non_open_spline_2())

        for spline in splines:
            ops = create_Lagrange_extraction_operators(spline)
            extraction_operators_tester(spline, evaluate_Lagrange_basis_1D, ops)


class TestCardinalExtractionOperators:
    """Test the create_cardinal_extraction_operators function."""

    def test_uniform_knot_vector(self):
        """Test cardinal extraction operators for uniform knot vector."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        spline = Bspline1D(knots, 2)
        result = create_cardinal_extraction_operators(spline)

        # Should have 4 intervals, 3x3 extraction matrices
        assert result.shape == (4, 3, 3)

        # Check that cardinal intervals have identity matrices
        cardinal_intervals = spline.get_cardinal_intervals()
        for i, is_cardinal in enumerate(cardinal_intervals):
            if is_cardinal:
                np.testing.assert_array_almost_equal(result[i], np.eye(3))

    def test_non_uniform_knot_vector(self):
        """Test cardinal extraction operators for non-uniform knot vector."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0]
        spline = Bspline1D(knots, 2)
        result = create_cardinal_extraction_operators(spline)

        # Should have 4 intervals, 3x3 extraction matrices
        assert result.shape == (4, 3, 3)

        # Check that some intervals might be cardinal (due to uniform spacing in some regions)
        cardinal_intervals = spline.get_cardinal_intervals()
        # Note: This knot vector has some uniform regions, so some intervals might be cardinal
        for i, is_cardinal in enumerate(cardinal_intervals):
            if is_cardinal:
                np.testing.assert_array_almost_equal(result[i], np.eye(3))

    def test_values(self):
        """Test the extraction operators."""
        n_intervals = 10
        splines = [
            create_uniform_spline(n_intervals, degree) for degree in [1, 2, 3, 4]
        ]
        splines.append(create_non_open_spline_1())
        splines.append(create_non_open_spline_2())

        for spline in splines:
            ops = create_cardinal_extraction_operators(spline)
            extraction_operators_tester(spline, evaluate_cardinal_Bspline_basis_1D, ops)


class TestIntegration:
    """Integration tests for change of basis operators."""

    def test_chain_of_transformations(self):
        """Test chain of basis transformations."""
        degree = 2

        # Create transformation matrices
        lagrange_to_bernstein = create_Lagrange_to_Bernstein_basis_operator(degree)
        bernstein_to_cardinal = create_Bernstein_to_cardinal_basis_operator(degree)

        # Chain transformation: Lagrange -> Bernstein -> Cardinal
        chain = bernstein_to_cardinal @ lagrange_to_bernstein

        # Should be invertible
        assert np.linalg.det(chain) != 0

    def test_consistency_with_different_degrees(self):
        """Test consistency across different degrees."""
        for degree in [1, 2, 3]:  # Skip degree 0 as basix doesn't support it
            # Test that all transformation matrices are invertible
            lagrange_to_bernstein = create_Lagrange_to_Bernstein_basis_operator(degree)
            bernstein_to_cardinal = create_Bernstein_to_cardinal_basis_operator(degree)

            assert np.linalg.det(lagrange_to_bernstein) != 0
            assert np.linalg.det(bernstein_to_cardinal) != 0

    def test_extraction_operators_consistency(self):
        """Test consistency of extraction operators."""
        knots = create_uniform_open_knot_vector(3, 2, start=0.0, end=1.0)
        spline = Bspline1D(knots, 2)

        # Test that all extraction operators have correct shape
        bezier_ops = create_Bezier_extraction_operators(spline)
        lagrange_ops = create_Lagrange_extraction_operators(spline)
        cardinal_ops = create_cardinal_extraction_operators(spline)

        num_intervals = spline.num_intervals
        degree = spline.degree

        assert bezier_ops.shape == (num_intervals, degree + 1, degree + 1)
        assert lagrange_ops.shape == (num_intervals, degree + 1, degree + 1)
        assert cardinal_ops.shape == (num_intervals, degree + 1, degree + 1)
