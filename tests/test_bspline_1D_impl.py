"""Tests for bspline_1D_impl module."""

import numpy as np
import pytest

from dolfinx_iga.splines.bspline_1D_impl import (
    _assert_spline_info,
    compute_num_basis_impl,
    create_bspline_Bezier_extraction_operators_impl,
    evaluate_basis_Cox_de_Boor_impl,
    get_cardinal_intervals_impl,
    get_last_knot_smaller_equal_impl,
    get_multiplicity_of_first_knot_in_domain_impl,
    get_unique_knots_and_multiplicity_impl,
    is_in_domain_impl,
)


class TestAssertSplineInfo:
    """Test the _assert_spline_info validation function."""

    def test_valid_inputs(self):
        """Test that valid inputs don't raise assertions."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        _assert_spline_info(knots, degree)

    def test_invalid_degree(self):
        """Test that negative degree raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = -1
        with pytest.raises(AssertionError, match="degree must be non-negative"):
            _assert_spline_info(knots, degree)

    def test_insufficient_knots(self):
        """Test that insufficient knots raise AssertionError."""
        knots = np.array([0.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="knots must have at least"):
            _assert_spline_info(knots, degree)

    def test_non_decreasing_knots(self):
        """Test that non-decreasing knots raise AssertionError."""
        knots = np.array([0.0, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="knots must be non-decreasing"):
            _assert_spline_info(knots, degree)


class TestGetMultiplicityOfFirstKnotInDomain:
    """Test the get_multiplicity_of_first_knot_in_domain_impl function."""

    def test_open_knot_vector(self):
        """Test multiplicity calculation for open knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        assert result == 3  # First knot in domain (index 2) has multiplicity 3

    def test_periodic_knot_vector(self):
        """Test multiplicity calculation for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        assert result == 1  # First knot in domain (index 2) has multiplicity 1

    def test_negative_tolerance_error(self):
        """Test that negative tolerance raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="tol must be positive"):
            get_multiplicity_of_first_knot_in_domain_impl(knots, degree, -1.0)


class TestGetUniqueKnotsAndMultiplicity:
    """Test the get_unique_knots_and_multiplicity_impl function."""

    def test_open_knot_vector_full(self):
        """Test unique knots extraction for full knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=False
        )
        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_open_knot_vector_domain_only(self):
        """Test unique knots extraction for domain only."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=True
        )
        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_periodic_knot_vector(self):
        """Test unique knots extraction for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=True
        )
        expected_unique = np.array([0.0, 0.5, 1.0])
        expected_mults = np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)


class TestIsInDomain:
    """Test the is_in_domain_impl function."""

    def test_points_in_domain(self):
        """Test that points within domain return True."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        tol = 1e-10
        result = is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [True, True, True])

    def test_points_outside_domain(self):
        """Test that points outside domain return False."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([-0.1, 1.1], dtype=np.float64)
        tol = 1e-10
        result = is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [False, False])

    def test_boundary_points(self):
        """Test that boundary points return True."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.0, 1.0], dtype=np.float64)
        tol = 1e-10
        result = is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [True, True])

    def test_empty_points_array(self):
        """Test that empty points array raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([], dtype=np.float64)
        tol = 1e-10
        with pytest.raises(AssertionError, match="pts must have at least one element"):
            is_in_domain_impl(knots, degree, pts, tol)


class TestComputeNumBasis:
    """Test the compute_num_basis_impl function."""

    def test_non_periodic_open(self):
        """Test basis count for non-periodic open knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        result = compute_num_basis_impl(knots, degree, periodic, tol)
        assert result == 3  # knots.size - degree - 1 = 6 - 2 - 1 = 3

    def test_periodic_knot_vector(self):
        """Test basis count for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        periodic = True
        tol = 1e-10
        result = compute_num_basis_impl(knots, degree, periodic, tol)
        # For periodic: num_basis = knots.size - degree - 1 - regularity - 1
        # regularity = degree - multiplicity_of_first_knot_in_domain
        # multiplicity_of_first_knot_in_domain = 1
        # regularity = 2 - 1 = 1
        # num_basis = 7 - 2 - 1 - 1 - 1 = 2
        assert result == 2

    def test_negative_tolerance_error(self):
        """Test that negative tolerance raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="tol must be positive"):
            compute_num_basis_impl(knots, degree, False, -1.0)


class TestGetLastKnotSmallerEqual:
    """Test the get_last_knot_smaller_equal_impl function."""

    def test_basic_functionality(self):
        """Test basic knot index finding."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)
        result = get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 2, 3])  # Indices of knots <= pts
        np.testing.assert_array_equal(result, expected)

    def test_knots_with_repetitions(self):
        """Test knots with repetitions index finding."""
        knots = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)
        result = get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 3, 4])  # Indices of knots <= pts
        np.testing.assert_array_equal(result, expected)

    def test_exact_knot_matches(self):
        """Test when points exactly match knots."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_non_decreasing_knots_error(self):
        """Test that non-decreasing knots raise AssertionError."""
        knots = np.array([0.0, 1.0, 0.5, 2.0], dtype=np.float64)
        pts = np.array([0.5], dtype=np.float64)
        with pytest.raises(AssertionError, match="knots must be non-decreasing"):
            get_last_knot_smaller_equal_impl(knots, pts)


class TestEvaluateBasisCoxDeBoor:
    """Test the evaluate_basis_Cox_de_Boor_impl function."""

    def test_bezier_like_evaluation(self):
        """Test evaluation for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis, first_basis = evaluate_basis_Cox_de_Boor_impl(
            knots, degree, periodic, tol, pts
        )

        # Check shape
        assert basis.shape == (3, 3)
        assert first_basis.shape == (3,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_general_knot_vector(self):
        """Test evaluation for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.25, 0.75], dtype=np.float64)

        basis, first_basis = evaluate_basis_Cox_de_Boor_impl(
            knots, degree, periodic, tol, pts
        )

        # Check shape
        assert basis.shape == (2, 3)
        assert first_basis.shape == (2,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_periodic_evaluation(self):
        """Test evaluation for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        periodic = True
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis, first_basis = evaluate_basis_Cox_de_Boor_impl(
            knots, degree, periodic, tol, pts
        )

        # Check shape
        assert basis.shape == (3, 3)
        assert first_basis.shape == (3,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_outside_domain_error(self):
        """Test that points outside domain raise AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([-0.1], dtype=np.float64)

        with pytest.raises(AssertionError):
            evaluate_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts)


class TestGetCardinalIntervals:
    """Test the get_cardinal_intervals_impl function."""

    def test_uniform_knot_vector(self):
        """Test cardinal intervals for uniform knot vector."""
        knots = np.array(
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], dtype=np.float64
        )
        degree = 2
        tol = 1e-10
        result = get_cardinal_intervals_impl(knots, degree, tol)

        # Should have 4 intervals, middle ones should be cardinal
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_non_uniform_knot_vector(self):
        """Test cardinal intervals for non-uniform knot vector."""
        knots = np.array(
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float64
        )
        degree = 2
        tol = 1e-10
        result = get_cardinal_intervals_impl(knots, degree, tol)

        # Should have 4 intervals, some might be cardinal due to uniform spacing in some regions
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_all_multiplicity_greater_than_one(self):
        """Test when all knots have multiplicity > 1."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = get_cardinal_intervals_impl(knots, degree, tol)

        # Should return all False
        expected = np.array([False])
        np.testing.assert_array_equal(result, expected)


class TestCreateBsplineBezierExtractionOperators:
    """Test the create_bspline_Bezier_extraction_operators_impl function."""

    def test_bezier_like_knot_vector(self):
        """Test extraction operators for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = create_bspline_Bezier_extraction_operators_impl(knots, degree, tol)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

        # For Bézier-like knots, extraction matrix should be identity
        np.testing.assert_array_almost_equal(result[0], np.eye(3))

    def test_general_knot_vector(self):
        """Test extraction operators for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = create_bspline_Bezier_extraction_operators_impl(knots, degree, tol)

        # Should have 2 intervals, 3x3 extraction matrices
        assert result.shape == (2, 3, 3)

        # Check that matrices are not identity (since not Bézier-like)
        assert not np.allclose(result[0], np.eye(3))
        assert not np.allclose(result[1], np.eye(3))

    def test_negative_tolerance_error(self):
        """Test that negative tolerance raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="tol must be positive"):
            create_bspline_Bezier_extraction_operators_impl(knots, degree, -1.0)
