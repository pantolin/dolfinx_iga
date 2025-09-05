"""Tests for bspline_1D module."""

import numpy as np
import pytest

from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.knots import (
    create_cardinal_Bspline_knot_vector,
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)
from dolfinx_iga.utils.tolerance import get_strict_tolerance


class TestBspline1DInit:
    """Test Bspline1D initialization."""

    def test_valid_initialization(self):
        """Test valid Bspline1D initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.degree == 2
        assert spline.periodic is False
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_zero_degree_initialization(self):
        """Test valid Bspline1D initialization."""
        knots = [0.0, 1.0]
        degree = 0
        spline = Bspline1D(knots, degree)

        assert spline.degree == 0
        assert spline.periodic is False
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_periodic_initialization(self):
        """Test periodic Bspline1D initialization."""
        knots = create_uniform_periodic_knot_vector(3, 2, start=0.0, end=1.0)
        degree = 2
        spline = Bspline1D(knots, degree, periodic=True)

        assert spline.degree == 2
        assert spline.periodic is True

    def test_integer_knots_conversion(self):
        """Test that integer knots are converted to float64."""
        knots = [0, 0, 0, 1, 1, 1]
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.dtype == np.float64
        np.testing.assert_array_equal(
            spline.knots, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        )

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="degree must be non-negative"):
            Bspline1D(knots, -1)

    def test_insufficient_knots_error(self):
        """Test that insufficient knots raise ValueError."""
        knots = [0.0, 1.0]
        with pytest.raises(ValueError, match="knots must have at least"):
            Bspline1D(knots, 2)

    def test_non_decreasing_knots_error(self):
        """Test that non-decreasing knots raise ValueError."""
        knots = [0.0, 1.0, 0.5, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="knots must be non-decreasing"):
            Bspline1D(knots, 2)

    def test_invalid_knot_type_error(self):
        """Test that invalid knot type raises TypeError."""
        knots = "invalid"
        with pytest.raises(
            (TypeError, ValueError),
            match="knots must be a 1D numpy array or Python list|knots type must be float",
        ):
            Bspline1D(knots, 2)

    def test_snap_knots_disabled(self):
        """Test initialization with snap_knots disabled."""
        tol = get_strict_tolerance(np.float64)
        knots = [0.0, 0.0, 0.0 + tol, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree, snap_knots=False)

        # Knots should remain unchanged
        np.testing.assert_array_equal(spline.knots, np.array(knots))


class TestBspline1DProperties:
    """Test Bspline1D properties."""

    def test_degree_property(self):
        """Test degree property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.degree == degree

    def test_knots_property(self):
        """Test knots property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_periodic_property(self):
        """Test periodic property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree, periodic=True)
        assert spline.periodic is True

    def test_tolerance_property(self):
        """Test tolerance property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.tolerance > 0

    def test_dtype_property(self):
        """Test dtype property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.dtype == np.float64


class TestBspline1DMethods:
    """Test Bspline1D methods."""

    def test_get_num_basis_non_periodic(self):
        """Test get_num_basis for non-periodic spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.get_num_basis() == 3

    def test_get_num_basis_periodic(self):
        """Test get_num_basis for periodic spline."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(3, degree, start=0.0, end=1.0)
        spline = Bspline1D(knots, degree, periodic=True)
        # For periodic splines, the number of basis functions is reduced
        assert spline.get_num_basis() == 3

    def test_get_unique_knots_and_multiplicity_full(self):
        """Test get_unique_knots_and_multiplicity for full knot vector."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        unique_knots, multiplicities = spline.get_unique_knots_and_multiplicity(
            in_domain=False
        )

        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_get_unique_knots_and_multiplicity_domain(self):
        """Test get_unique_knots_and_multiplicity for domain only."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        unique_knots, multiplicities = spline.get_unique_knots_and_multiplicity(
            in_domain=True
        )

        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_get_num_intervals(self):
        """Test get_num_intervals method."""
        num_intervals = 2
        degree = 2
        knots = create_uniform_open_knot_vector(num_intervals, degree)
        spline = Bspline1D(knots, degree)
        assert spline.get_num_intervals() == num_intervals

    def test_get_domain(self):
        """Test get_domain method."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        domain = spline.get_domain()
        assert domain == (knots[degree], knots[-degree - 1])

    def test_has_left_end_open_true(self):
        """Test has_left_end_open returns True for open left end."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_left_end_open() is True

    def test_has_left_end_open_false(self):
        """Test has_left_end_open returns False for non-open left end."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_left_end_open() is False

    def test_has_right_end_open_true(self):
        """Test has_right_end_open returns True for open right end."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_right_end_open() is True

    def test_has_right_end_open_false(self):
        """Test has_right_end_open returns False for non-open right end."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_right_end_open() is False

    def test_has_open_knots_true(self):
        """Test has_open_knots returns True when both ends are open."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_open_knots() is True

    def test_has_open_knots_false(self):
        """Test has_open_knots returns False when ends are not open."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_open_knots() is False

    def test_has_Bezier_like_knots_true(self):
        """Test has_Bezier_like_knots returns True for Bézier-like configuration."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_has_Bezier_like_knots_false(self):
        """Test has_Bezier_like_knots returns False for non-Bézier-like configuration."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert bool(spline.has_Bezier_like_knots()) is False

    def test_has_Bezier_like_knots_periodic_false(self):
        """Test has_Bezier_like_knots returns False for periodic splines."""
        # Use a valid periodic knot vector
        degree = 2
        knots = create_uniform_periodic_knot_vector(3, degree, start=0.0, end=1.0)
        spline = Bspline1D(knots, degree, periodic=True)
        assert bool(spline.has_Bezier_like_knots()) is False

    def test_get_cardinal_intervals(self):
        """Test get_cardinal_intervals method."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        result = spline.get_cardinal_intervals()

        # Should have 4 intervals, middle ones should be cardinal
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)


class TestBspline1DWithKnotGenerators:
    """Test Bspline1D with knot vector generators."""

    def test_with_uniform_open_knot_vector(self):
        """Test Bspline1D with uniform open knot vector."""
        knots = create_uniform_open_knot_vector(2, 2, start=0.0, end=1.0)
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.has_open_knots() is True
        assert spline.get_domain() == (knots[degree], knots[-degree - 1])

    def test_with_uniform_periodic_knot_vector(self):
        """Test Bspline1D with uniform periodic knot vector."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(3, degree, start=0.0, end=1.0)
        spline = Bspline1D(knots, degree, periodic=True)

        assert spline.degree == degree
        assert spline.periodic is True
        assert spline.get_domain() == (knots[degree], knots[-degree - 1])

    def test_with_cardinal_bspline_knot_vector(self):
        """Test Bspline1D with cardinal B-spline knot vector."""
        degree = 2
        knots = create_cardinal_Bspline_knot_vector(2, degree)
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.get_domain() == (knots[degree], knots[-degree - 1])


class TestBspline1DEdgeCases:
    """Test Bspline1D edge cases."""

    def test_degree_zero(self):
        """Test Bspline1D with degree 0."""
        knots = [0.0, 1.0]
        degree = 0
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.get_num_basis() == 1
        assert spline.get_domain() == (knots[degree], knots[-degree - 1])

    def test_single_interval(self):
        """Test Bspline1D with single interval."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.get_num_intervals() == 1
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_high_degree(self):
        """Test Bspline1D with high degree."""
        degree = 5
        knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.get_num_basis() == (degree + 1)
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_float32_precision(self):
        """Test Bspline1D with float32 precision."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.dtype == np.float32
        assert spline.tolerance > 0


class TestBspline1DIntegration:
    """Integration tests for Bspline1D."""

    def test_consistency_across_methods(self):
        """Test consistency across different Bspline1D methods."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)

        # Test that domain indices are consistent
        domain = spline.get_domain()
        unique_knots, _ = spline.get_unique_knots_and_multiplicity(in_domain=True)

        assert domain[0] == unique_knots[0]
        np.testing.assert_array_almost_equal(domain[1], unique_knots[-1])

        # Test that number of intervals is consistent
        num_intervals = spline.get_num_intervals()
        assert num_intervals == len(unique_knots) - 1

    def test_periodic_vs_non_periodic_consistency(self):
        """Test consistency between periodic and non-periodic versions."""
        # Create equivalent knot vectors
        degree = 2
        knots_open = create_uniform_open_knot_vector(3, degree, start=0.0, end=1.0)
        knots_periodic = create_uniform_periodic_knot_vector(
            3, degree, start=0.0, end=1.0
        )

        spline_open = Bspline1D(knots_open, degree, periodic=False)
        spline_periodic = Bspline1D(knots_periodic, degree, periodic=True)

        # Both should have the same domain
        assert spline_open.get_domain() == spline_periodic.get_domain()

        # Both should have the same number of intervals
        assert spline_open.get_num_intervals() == spline_periodic.get_num_intervals()

        # But different number of basis functions
        assert spline_open.get_num_basis() != spline_periodic.get_num_basis()

    def test_knot_snapping_consistency(self):
        """Test that knot snapping doesn't break consistency."""
        # Create knots with small numerical differences
        knots = [0.0, 0.0, 0.0, 0.5000000001, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree, snap_knots=True)

        # After snapping, should still be valid
        assert spline.get_num_basis() > 0
        assert spline.get_num_intervals() > 0

        # Domain should be well-defined
        domain = spline.get_domain()
        assert domain[0] < domain[1]
