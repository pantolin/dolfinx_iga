"""Tests for knots module."""

import numpy as np
import pytest

from dolfinx_iga.knots import KnotsVector, create_open_uniform_knot_vector


class TestKnotsVector:
    """Test KnotsVector class functionality."""

    def test_init_valid_knots(self):
        """Test KnotsVector initialization with valid knots."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64)
        kv = KnotsVector(knots)

        assert len(kv) == 5
        assert kv.dtype == np.float64
        np.testing.assert_allclose(kv.knots_with_repetitions, knots, rtol=1e-14)

    def test_init_with_snapping(self):
        """Test KnotsVector initialization with automatic snapping."""
        knots = np.array([0.0, 1e-8, 0.5, 1.0, 1.0 + 1e-8], dtype=np.float32)
        kv = KnotsVector(knots, snap_knots=True)

        # Values close to tolerance should be snapped
        assert kv[0] == kv[1]  # 0.0 and 1e-8 should be equal
        assert kv[3] == kv[4]  # 1.0 and 1.0 + 1e-8 should be equal

    def test_init_without_snapping(self):
        """Test KnotsVector initialization without snapping."""
        knots = np.array([0.0, 0.0000001, 0.5, 1.0, 1.0000001], dtype=np.float32)
        kv = KnotsVector(knots, snap_knots=False)

        # Values should remain as-is
        np.testing.assert_array_equal(kv.knots_with_repetitions, knots)

    def test_init_custom_tolerance(self):
        """Test KnotsVector initialization with custom tolerance."""
        knots = np.array([0.0, 0.001, 0.5, 1.0, 1.001], dtype=np.float64)
        kv = KnotsVector(knots, snap_knots=True, tolerance=np.float64(0.01))

        # With large tolerance, 0.001 differences should be snapped
        assert kv[0] == kv[1]
        assert kv[3] == kv[4]

    def test_init_invalid_dimension(self):
        """Test KnotsVector initialization with invalid dimensions."""
        # 2D array
        knots_2d = np.array([[0.0, 0.5], [1.0, 1.5]])
        with pytest.raises(ValueError, match="1D numpy array"):
            KnotsVector(knots_2d)

        # Single element
        knots_single = np.array([1.0])
        with pytest.raises(ValueError, match="at least two elements"):
            KnotsVector(knots_single)

        # Empty array
        knots_empty = np.array([])
        with pytest.raises(ValueError, match="at least two elements"):
            KnotsVector(knots_empty)

    def test_init_non_monotonic(self):
        """Test KnotsVector initialization with non-monotonic knots."""
        knots = np.array([0.0, 1.0, 0.5, 2.0])  # Non-monotonic
        with pytest.raises(ValueError, match="monotonically increasing"):
            KnotsVector(knots)

    def test_init_no_nonzero_spans(self):
        """Test KnotsVector initialization that results in no nonzero spans."""
        knots = np.array([1.0, 1.0, 1.0, 1.0])  # All same value
        with pytest.raises(ValueError, match="at least one non zero span"):
            KnotsVector(knots)

    def test_len_and_getitem(self):
        """Test __len__ and __getitem__ methods."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv = KnotsVector(knots)

        assert len(kv) == 5
        assert kv[0] == 0.0
        assert np.isclose(kv[2], 0.5)
        assert np.isclose(kv[4], 1.0)

    def test_repr_and_str(self):
        """Test string representations."""
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Test repr
        repr_str = repr(kv)
        assert "KnotsVector" in repr_str

        # Test str
        str_str = str(kv)
        assert "Knots with repetitions" in str_str
        assert "Unique knots" in str_str
        assert "Multiplicities" in str_str

    def test_properties(self):
        """Test property methods."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
        kv = KnotsVector(knots)

        # Test knots_with_repetitions
        np.testing.assert_array_equal(kv.knots_with_repetitions, knots)

        # Test unique_knots
        expected_unique = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(kv.unique_knots, expected_unique)

        # Test multiplicities
        expected_multiplicities = np.array([2, 1, 2])
        np.testing.assert_array_equal(kv.multiplicities, expected_multiplicities)

        # Test dtype
        assert kv.dtype == np.float32

    def test_num_nonzero_spans(self):
        """Test num_nonzero_spans method."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv = KnotsVector(knots)

        assert kv.num_nonzero_spans() == 2  # [0,0.5] and [0.5,1]

    def test_is_open(self):
        """Test is_open method."""
        # Create open knot vector for degree 2
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])  # 3 repetitions at ends
        kv = KnotsVector(knots)

        assert kv.is_open(degree=2)  # Should be open for degree 2
        assert not kv.is_open(degree=1)  # Should not be open for degree 1
        assert not kv.is_open(degree=3)  # Should not be open for degree 3

    def test_is_open_invalid_degree(self):
        """Test is_open with invalid degree."""
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        kv = KnotsVector(knots)

        with pytest.raises(ValueError, match="non-negative"):
            kv.is_open(degree=-1)

    def test_is_uniform(self):
        """Test is_uniform method."""
        # Uniform knot vector
        uniform_knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv_uniform = KnotsVector(uniform_knots)
        assert kv_uniform.is_uniform()

        # Non-uniform knot vector
        non_uniform_knots = np.array([0.0, 0.0, 0.3, 1.0, 1.0])
        kv_non_uniform = KnotsVector(non_uniform_knots)
        assert not kv_non_uniform.is_uniform()

    def test_find_span_basic(self):
        """Test basic find_span functionality."""
        # Create knot vector: [0, 0, 0.5, 1, 1] with spans [0, 0.5) and [0.5, 1]
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Test values in first span [0, 0.5)
        assert kv.find_span(np.array([0.0])) == 0
        assert kv.find_span(np.array([0.25])) == 0
        assert kv.find_span(np.array([0.49])) == 0

        # Test values in second span [0.5, 1]
        assert kv.find_span(np.array([0.5])) == 1
        assert kv.find_span(np.array([0.75])) == 1
        assert kv.find_span(np.array([1.0])) == 1  # Special case: last knot

    def test_find_span_multiple_values(self):
        """Test find_span with multiple values at once."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv = KnotsVector(knots)

        values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected_spans = np.array([0, 0, 1, 1, 1])

        spans = kv.find_span(values)
        np.testing.assert_array_equal(spans, expected_spans)

    def test_find_span_three_spans(self):
        """Test find_span with three spans."""
        # Create knot vector with three spans: [0, 0.33), [0.33, 0.67), [0.67, 1]
        knots = np.array([0.0, 0.0, 0.33, 0.67, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Test boundary and interior values
        test_values = np.array([0.0, 0.1, 0.33, 0.5, 0.67, 0.8, 1.0])
        expected_spans = np.array([0, 0, 1, 1, 2, 2, 2])

        spans = kv.find_span(test_values)
        np.testing.assert_array_equal(spans, expected_spans)

    def test_find_span_repeated_interior_knots(self):
        """Test find_span with repeated interior knots."""
        # Knot vector: [0, 0, 0, 0.5, 0.5, 1, 1, 1] with spans [0, 0.5) and [0.5, 1]
        knots = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Test values around the repeated interior knot
        test_values = np.array([0.0, 0.25, 0.49, 0.5, 0.51, 0.75, 1.0])
        expected_spans = np.array([0, 0, 0, 1, 1, 1, 1])

        spans = kv.find_span(test_values)
        np.testing.assert_array_equal(spans, expected_spans)

    def test_find_span_boundary_tolerance(self):
        """Test find_span with values at boundaries within tolerance."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64)
        kv = KnotsVector(knots)

        # Test values slightly outside bounds but within tolerance
        tol = 1e-16  # Use a very small tolerance (smaller than actual tolerance)

        # Values slightly before start (should be clamped to first span)
        slightly_before = np.array([-tol / 2])
        assert kv.find_span(slightly_before) == 0

        # Values slightly after end (should be clamped to last span)
        slightly_after = np.array([1.0 + tol / 2])
        assert kv.find_span(slightly_after) == 1

    def test_find_span_single_value_input(self):
        """Test find_span with single scalar value."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Test with scalar input (wrapped in array) - should return array
        span = kv.find_span(np.array([0.25]))
        assert span == 0
        assert isinstance(span, np.ndarray)

        span = kv.find_span(np.array([0.75]))
        assert span == 1
        assert isinstance(span, np.ndarray)

    def test_find_span_dtype_preservation(self):
        """Test that find_span returns integer array."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
        kv = KnotsVector(knots)

        values = np.array([0.25, 0.75], dtype=np.float32)
        spans = kv.find_span(values)

        assert spans.dtype == np.int_
        np.testing.assert_array_equal(spans, [0, 1])

    def test_find_span_out_of_bounds_error(self):
        """Test find_span with values outside knot vector domain."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64)
        kv = KnotsVector(knots)

        # Values significantly outside the tolerance should raise ValueError
        with pytest.raises(ValueError, match="must be within the knot vector domain"):
            kv.find_span(np.array([-0.1]))  # Too far before start

        with pytest.raises(ValueError, match="must be within the knot vector domain"):
            kv.find_span(np.array([1.1]))  # Too far after end

        with pytest.raises(ValueError, match="must be within the knot vector domain"):
            kv.find_span(np.array([0.5, 1.1]))  # Mixed valid/invalid

    def test_find_span_edge_cases(self):
        """Test find_span edge cases."""
        # Minimal knot vector with just two unique knots
        knots = np.array([0.0, 1.0])
        kv = KnotsVector(knots)

        # Only one span [0, 1]
        assert kv.find_span(np.array([0.0])) == 0
        assert kv.find_span(np.array([0.5])) == 0
        assert kv.find_span(np.array([1.0])) == 0

    def test_find_span_uniform_spacing(self):
        """Test find_span with uniformly spaced knots."""
        # Create uniform knot vector [0, 0, 0.25, 0.5, 0.75, 1, 1]
        # Spans: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]
        knots = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Test systematic coverage of all spans
        test_values = np.array([0.0, 0.1, 0.25, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0])
        expected_spans = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])

        spans = kv.find_span(test_values)
        np.testing.assert_array_equal(spans, expected_spans)

    def test_find_span_large_array(self):
        """Test find_span with a large array of values."""
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        kv = KnotsVector(knots)

        # Create many test values
        test_values = np.linspace(0.0, 1.0, 1000)
        spans = kv.find_span(test_values)

        # All values in [0, 0.5) should map to span 0
        first_half_mask = test_values < 0.5
        assert np.all(spans[first_half_mask] == 0)

        # All values in [0.5, 1] should map to span 1
        second_half_mask = test_values >= 0.5
        assert np.all(spans[second_half_mask] == 1)


class TestCreateOpenUniformKnotVector:
    """Test create_open_uniform_knot_vector function."""

    def test_basic_creation(self):
        """Test basic open uniform knot vector creation."""
        kv = create_open_uniform_knot_vector(
            degree=2, start=0.0, end=1.0, num_intervals=2
        )

        # Should create: [0,0,0, 0.5, 1,1,1]
        # For degree 2, with default continuity=1, interior_multiplicity = 2-1 = 1
        # Boundary knots get degree+1 = 3 repetitions, interior knots get 1
        expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(kv.knots_with_repetitions, expected)

        # Should be open for the specified degree
        assert kv.is_open(degree=2)

    def test_different_degrees(self):
        """Test creation with different degrees."""
        for degree in [1, 2, 3, 4]:
            kv = create_open_uniform_knot_vector(
                degree=degree, start=0.0, end=1.0, num_intervals=1
            )

            # Check that it's open for the specified degree
            assert kv.is_open(degree=degree)

            # Check correct multiplicities at ends
            assert kv.multiplicities[0] == degree + 1  # First knot
            assert kv.multiplicities[-1] == degree + 1  # Last knot

    def test_custom_continuity(self):
        """Test creation with custom continuity."""
        # Degree 3 with C^1 continuity (instead of default C^2)
        kv = create_open_uniform_knot_vector(
            degree=3, start=0.0, end=1.0, num_intervals=2, continuity=1
        )

        # Interior knots should have multiplicity = degree - continuity = 3 - 1 = 2
        # Expected: [0,0,0,0, 0,0, 0.5,0.5, 1,1, 1,1,1,1]
        assert kv.multiplicities[1] == 2  # Interior knot multiplicity

    def test_c_minus_1_continuity(self):
        """Test creation with C^(-1) continuity (discontinuous)."""
        kv = create_open_uniform_knot_vector(
            degree=2, start=0.0, end=1.0, num_intervals=2, continuity=-1
        )

        # Interior knots should have multiplicity = degree - (-1) = 2 + 1 = 3
        assert kv.multiplicities[1] == 3  # Interior knot has full multiplicity

    def test_custom_dtype(self):
        """Test creation with custom dtype."""
        kv = create_open_uniform_knot_vector(
            degree=1, start=0.0, end=1.0, num_intervals=1, dtype=np.dtype(np.float32)
        )

        assert kv.dtype == np.float32

    def test_custom_range(self):
        """Test creation with custom start/end range."""
        kv = create_open_uniform_knot_vector(
            degree=1, start=-1.0, end=2.0, num_intervals=3
        )

        # Should span from -1 to 2
        assert kv.knots_with_repetitions[0] == -1.0
        assert kv.knots_with_repetitions[-1] == 2.0

        # Check uniform spacing
        assert kv.is_uniform()

    def test_invalid_degree(self):
        """Test creation with invalid degree."""
        with pytest.raises(ValueError, match="non-negative"):
            create_open_uniform_knot_vector(
                degree=-1, start=0.0, end=1.0, num_intervals=1
            )

    def test_invalid_range(self):
        """Test creation with invalid range."""
        with pytest.raises(ValueError, match="Start must be less than end"):
            create_open_uniform_knot_vector(
                degree=1, start=1.0, end=0.0, num_intervals=1
            )

        with pytest.raises(ValueError, match="Start must be less than end"):
            create_open_uniform_knot_vector(
                degree=1, start=1.0, end=1.0, num_intervals=1
            )

    def test_invalid_num_intervals(self):
        """Test creation with invalid number of intervals."""
        with pytest.raises(ValueError, match="at least 1"):
            create_open_uniform_knot_vector(
                degree=1, start=0.0, end=1.0, num_intervals=0
            )

    def test_invalid_continuity(self):
        """Test creation with invalid continuity."""
        # Continuity too high
        with pytest.raises(ValueError, match="Continuity must be between"):
            create_open_uniform_knot_vector(
                degree=2, start=0.0, end=1.0, num_intervals=1, continuity=3
            )

        # Continuity too low
        with pytest.raises(ValueError, match="Continuity must be between"):
            create_open_uniform_knot_vector(
                degree=2, start=0.0, end=1.0, num_intervals=1, continuity=-2
            )

    def test_multiple_intervals(self):
        """Test creation with multiple intervals."""
        kv = create_open_uniform_knot_vector(
            degree=2, start=0.0, end=1.0, num_intervals=4
        )

        # Should have 4 intervals: [0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]
        assert kv.num_nonzero_spans() == 4

        # Check uniform spacing
        assert kv.is_uniform()

        # Unique knots should be [0, 0.25, 0.5, 0.75, 1.0]
        expected_unique = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(kv.unique_knots, expected_unique)
