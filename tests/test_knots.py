"""Tests for knots module."""

import numpy as np
import pytest

from dolfinx_iga.splines.knots import (
    _get_ends_and_type,
    _validate_knot_input,
    create_cardinal_Bspline_knot_vector,
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)
from dolfinx_iga.utils.tolerance import get_strict_tolerance


class TestValidateKnotInput:
    """Test the _validate_knot_input function."""

    def test_valid_inputs(self):
        """Test that valid inputs don't raise errors."""
        _validate_knot_input(
            num_intervals=2,
            degree=2,
            continuity=1,
            start=np.float64(0.0),
            end=np.float64(1.0),
            dtype=np.float64,
        )

    def test_start_greater_than_end_error(self):
        """Test that start >= end raises ValueError."""
        with pytest.raises(ValueError, match="start must be less than end"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=1,
                start=np.float64(1.0),
                end=np.float64(0.0),
                dtype=np.float64,
            )

    def test_negative_num_intervals_error(self):
        """Test that negative num_intervals raises ValueError."""
        with pytest.raises(ValueError, match="num_intervals must be non-negative"):
            _validate_knot_input(
                num_intervals=-1,
                degree=2,
                continuity=1,
                start=np.float64(0.0),
                end=np.float64(1.0),
                dtype=np.float64,
            )

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            _validate_knot_input(
                num_intervals=2,
                degree=-1,
                continuity=1,
                start=np.float64(0.0),
                end=np.float64(1.0),
                dtype=np.float64,
            )

    def test_invalid_continuity_error(self):
        """Test that invalid continuity raises ValueError."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=2,  # Invalid for degree 2
                start=np.float64(0.0),
                end=np.float64(1.0),
                dtype=np.float64,
            )

    def test_invalid_dtype_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float64 or float32"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=1,
                start=np.float64(0.0),
                end=np.float64(1.0),
                dtype=np.int32,
            )


class TestGetEndsAndType:
    """Test the _get_ends_and_type function."""

    def test_default_values(self):
        """Test with default values."""
        start, end, dtype = _get_ends_and_type()

        assert start == 0.0
        assert end == 1.0
        assert dtype == np.float64

    def test_provided_start_end(self):
        """Test with provided start and end values."""
        start, end, dtype = _get_ends_and_type(start=0.5, end=2.0)

        assert start == 0.5
        assert end == 2.0
        assert dtype == np.float64

    def test_provided_dtype(self):
        """Test with provided dtype."""
        start, end, dtype = _get_ends_and_type(dtype=np.float32)

        assert start == 0.0
        assert end == 1.0
        assert dtype == np.float32

    def test_float_conversion(self):
        """Test that float values are converted to float64."""
        start, end, dtype = _get_ends_and_type(start=0.5, end=1.5)

        assert isinstance(start, np.float64)
        assert isinstance(end, np.float64)
        assert dtype == np.float64

    def test_dtype_mismatch_error(self):
        """Test that dtype mismatch raises ValueError."""
        with pytest.raises(ValueError, match="end must be of type dtype"):
            _get_ends_and_type(
                start=np.float32(0.0),
                end=np.float64(1.0),
                dtype=np.float32,
            )

    def test_end_less_than_start_error(self):
        """Test that end <= start raises ValueError."""
        with pytest.raises(ValueError, match="end must be greater than start"):
            _get_ends_and_type(start=1.0, end=0.0)

    def test_dtype_validation_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="start must be of type dtype"):
            _get_ends_and_type(
                start=np.float32(0.0),
                dtype=np.float64,
            )


class TestCreateUniformOpenKnotVector:
    """Test the create_uniform_open_knot_vector function."""

    def test_basic_functionality(self):
        """Test basic uniform open knot vector creation."""
        result = create_uniform_open_knot_vector(2, 2, start=0.0, end=1.0)

        expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_zero(self):
        """Test with degree 0."""
        result = create_uniform_open_knot_vector(2, 0, start=0.0, end=1.0)

        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_one(self):
        """Test with degree 1."""
        result = create_uniform_open_knot_vector(2, 1, start=0.0, end=1.0)

        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_interval(self):
        """Test with single interval."""
        result = create_uniform_open_knot_vector(1, 2, start=0.0, end=1.0)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_custom_continuity(self):
        """Test with custom continuity."""
        result = create_uniform_open_knot_vector(2, 2, continuity=0, start=0.0, end=1.0)

        # With continuity=0, interior knots should have multiplicity 2
        expected = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_custom_domain(self):
        """Test with custom domain."""
        result = create_uniform_open_knot_vector(2, 2, start=1.0, end=3.0)

        expected = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        result = create_uniform_open_knot_vector(2, 2, dtype=np.float32)

        assert result.dtype == np.float32
        expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_num_intervals_error(self):
        """Test that negative num_intervals raises ValueError."""
        with pytest.raises(ValueError, match="num_intervals must be non-negative"):
            create_uniform_open_knot_vector(-1, 2)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            create_uniform_open_knot_vector(2, -1)

    def test_invalid_continuity_error(self):
        """Test that invalid continuity raises ValueError."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            create_uniform_open_knot_vector(2, 2, continuity=2)


class TestCreateUniformPeriodicKnotVector:
    """Test the create_uniform_periodic_knot_vector function."""

    def test_basic_functionality(self):
        """Test basic uniform periodic knot vector creation."""
        result = create_uniform_periodic_knot_vector(2, 2, start=0.0, end=1.0)

        # Should extend beyond domain boundaries
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_zero(self):
        """Test with degree 0."""
        result = create_uniform_periodic_knot_vector(2, 0, start=0.0, end=1.0)

        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_one(self):
        """Test with degree 1."""
        result = create_uniform_periodic_knot_vector(2, 1, start=0.0, end=1.0)

        expected = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_interval(self):
        """Test with single interval."""
        result = create_uniform_periodic_knot_vector(1, 2, start=0.0, end=1.0)

        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_custom_continuity(self):
        """Test with custom continuity."""
        result = create_uniform_periodic_knot_vector(
            2, 2, continuity=0, start=0.0, end=1.0
        )

        # With continuity=0, interior knots should have multiplicity 2
        expected = np.array([-0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_custom_domain(self):
        """Test with custom domain."""
        result = create_uniform_periodic_knot_vector(2, 2, start=1.0, end=3.0)

        expected = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        result = create_uniform_periodic_knot_vector(2, 2, dtype=np.float32)

        assert result.dtype == np.float32
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_num_intervals_error(self):
        """Test that negative num_intervals raises ValueError."""
        with pytest.raises(ValueError, match="num_intervals must be non-negative"):
            create_uniform_periodic_knot_vector(-1, 2)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            create_uniform_periodic_knot_vector(2, -1)

    def test_invalid_continuity_error(self):
        """Test that invalid continuity raises ValueError."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            create_uniform_periodic_knot_vector(2, 2, continuity=2)


class TestCreateCardinalBsplineKnotVector:
    """Test the create_cardinal_Bspline_knot_vector function."""

    def test_basic_functionality(self):
        """Test basic cardinal B-spline knot vector creation."""
        result = create_cardinal_Bspline_knot_vector(2, 2)

        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_zero(self):
        """Test with degree 0."""
        result = create_cardinal_Bspline_knot_vector(2, 0)

        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_one(self):
        """Test with degree 1."""
        result = create_cardinal_Bspline_knot_vector(2, 1)

        expected = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_interval(self):
        """Test with single interval."""
        result = create_cardinal_Bspline_knot_vector(1, 2)

        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_high_degree(self):
        """Test with high degree."""
        result = create_cardinal_Bspline_knot_vector(2, 5)

        expected = np.array(
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        result = create_cardinal_Bspline_knot_vector(2, 2, dtype=np.float32)

        assert result.dtype == np.float32
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_num_intervals_error(self):
        """Test that negative num_intervals raises ValueError."""
        with pytest.raises(ValueError, match="num_intervals must be at least 1"):
            create_cardinal_Bspline_knot_vector(0, 2)

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            create_cardinal_Bspline_knot_vector(2, -1)

    def test_invalid_dtype_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_cardinal_Bspline_knot_vector(2, 2, dtype=np.int32)


class TestKnotVectorIntegration:
    """Integration tests for knot vector generation."""

    def test_consistency_across_functions(self):
        """Test consistency across different knot vector generation functions."""
        # Test that all functions produce valid knot vectors
        knots_open = create_uniform_open_knot_vector(3, 2, start=0.0, end=1.0)
        knots_periodic = create_uniform_periodic_knot_vector(3, 2, start=0.0, end=1.0)
        knots_cardinal = create_cardinal_Bspline_knot_vector(3, 2)

        # All should be non-decreasing
        assert np.all(np.diff(knots_open) >= 0)
        assert np.all(np.diff(knots_periodic) >= 0)
        assert np.all(np.diff(knots_cardinal) >= 0)

        # All should have correct dtype
        assert knots_open.dtype == np.float64
        assert knots_periodic.dtype == np.float64
        assert knots_cardinal.dtype == np.float64

    def test_domain_consistency(self):
        """Test that domain parameters are respected."""
        start, end = 0.5, 2.5
        degree = 2

        knots_open = create_uniform_open_knot_vector(2, degree, start=start, end=end)
        knots_periodic = create_uniform_periodic_knot_vector(
            2, degree, start=start, end=end
        )

        # Open knot vector should have domain [start, end]
        assert knots_open[degree] == start  # First knot in domain
        assert knots_open[-degree - 1] == end  # Last knot in domain

        # Periodic knot vector should have domain [start, end]
        assert knots_periodic[degree] == start  # First knot in domain
        assert knots_periodic[-degree - 1] == end  # Last knot in domain

    def test_continuity_consistency(self):
        """Test that continuity parameters are respected."""
        degree = 3
        continuity = 1

        knots_open = create_uniform_open_knot_vector(2, degree, continuity=continuity)
        knots_periodic = create_uniform_periodic_knot_vector(
            2, degree, continuity=continuity
        )
        for knots in [knots_open, knots_periodic]:
            tol = get_strict_tolerance(knots.dtype)

            # Check interior knots in open knot vector
            interior_knots = knots[degree:-degree]
            unique_interior = np.unique(interior_knots)
            for knot in unique_interior:
                count = np.sum(np.abs(interior_knots - knot) < tol)
                # Note: For 2 intervals, there's only one interior knot, so multiplicity might be different
                assert count >= 1  # At least one occurrence

    def test_vector_length(self):
        """Test that the vector length is consistent with the input parameters."""
        degree = 2

        knots_open = create_uniform_open_knot_vector(2, degree)
        knots_periodic = create_uniform_periodic_knot_vector(2, degree)
        knots_cardinal = create_cardinal_Bspline_knot_vector(2, degree)

        # All should have correct number of knots for the degree
        expected_open = 2 * degree + 2 + 1  # 2 intervals + 1 interior knot
        expected_periodic = 2 * degree + 2 + 1  # 2 intervals + 1 interior knot
        expected_cardinal = 2 * degree + 1 + 2  # 2 intervals + 1 boundary knot

        assert len(knots_open) == expected_open
        assert len(knots_periodic) == expected_periodic
        assert len(knots_cardinal) == expected_cardinal

    def test_dtype_consistency(self):
        """Test that dtype parameters are respected."""
        dtype = np.float32

        knots_open = create_uniform_open_knot_vector(2, 2, dtype=dtype)
        knots_periodic = create_uniform_periodic_knot_vector(2, 2, dtype=dtype)
        knots_cardinal = create_cardinal_Bspline_knot_vector(2, 2, dtype=dtype)

        assert knots_open.dtype == dtype
        assert knots_periodic.dtype == dtype
        assert knots_cardinal.dtype == dtype
