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
        np.testing.assert_array_equal(kv.knots_with_repetitions, knots)

    def test_init_with_snapping(self):
        """Test KnotsVector initialization with automatic snapping."""
        knots = np.array([0.0, 0.0000001, 0.5, 1.0, 1.0000001], dtype=np.float32)
        kv = KnotsVector(knots, snap_knots=True)
        
        # Values close to tolerance should be snapped
        assert kv[0] == kv[1]  # 0.0 and 0.0000001 should be equal
        assert kv[3] == kv[4]  # 1.0 and 1.0000001 should be equal

    def test_init_without_snapping(self):
        """Test KnotsVector initialization without snapping."""
        knots = np.array([0.0, 0.0000001, 0.5, 1.0, 1.0000001], dtype=np.float32)
        kv = KnotsVector(knots, snap_knots=False)
        
        # Values should remain as-is
        np.testing.assert_array_equal(kv.knots_with_repetitions, knots)

    def test_init_custom_snap_tolerance(self):
        """Test KnotsVector initialization with custom snap tolerance."""
        knots = np.array([0.0, 0.001, 0.5, 1.0, 1.001], dtype=np.float64)
        kv = KnotsVector(knots, snap_knots=True, snap_tolerance=0.01)
        
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
        assert kv[2] == 0.5
        assert kv[4] == 1.0

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

    def test_snap_to_tolerance_static_method(self):
        """Test snap_to_tolerance static method."""
        # Create a knot vector with values that should be snapped together
        knots = np.array([0.0, 0.0000001, 1.0, 1.0000001, 2.0], dtype=np.float32)
        
        # Use default tolerance
        snapped = KnotsVector.snap_to_tolerance(knots)
        
        # With float32 default tolerance (1e-6), differences of 1e-7 should be snapped
        assert snapped[0] == snapped[1]  # 0.0 and 0.0000001 should be equal
        assert snapped[2] == snapped[3]  # 1.0 and 1.0000001 should be equal
        assert snapped[4] == 2.0  # 2.0 should remain unchanged
        
    def test_snap_to_tolerance_custom_tolerance(self):
        """Test snap_to_tolerance with custom tolerance."""
        knots = np.array([0.0, 0.001, 1.0, 1.001, 2.0], dtype=np.float64)
        
        # Use a large tolerance that should snap 0.001 differences
        tol = 0.01
        snapped = KnotsVector.snap_to_tolerance(knots, tol=tol)
        
        # With tolerance 0.01, differences of 0.001 should be snapped
        assert snapped[0] == snapped[1]  # 0.0 and 0.001 should be equal
        assert snapped[2] == snapped[3]  # 1.0 and 1.001 should be equal
        assert snapped[4] == 2.0  # 2.0 should remain unchanged
        
    def test_snap_to_tolerance_strict_tolerance(self):
        """Test snap_to_tolerance with strict tolerance that doesn't snap values."""
        knots = np.array([0.0, 0.001, 1.0, 1.001, 2.0], dtype=np.float64)
        
        # Use a very small tolerance that shouldn't snap 0.001 differences  
        tol = 1e-6
        snapped = KnotsVector.snap_to_tolerance(knots, tol=tol)
        
        # With strict tolerance, values should remain different
        assert snapped[0] != snapped[1]  # 0.0 and 0.001 should remain different
        assert snapped[2] != snapped[3]  # 1.0 and 1.001 should remain different
        assert np.allclose(snapped, knots)  # Should be essentially unchanged
        
    def test_snap_to_tolerance_preserves_order(self):
        """Test that snap_to_tolerance preserves knot order."""
        knots = np.array([0.0, 0.0000001, 0.5, 1.0, 1.0000001, 2.0], dtype=np.float32)
        snapped = KnotsVector.snap_to_tolerance(knots)
        
        # Order should be preserved
        assert np.all(snapped[:-1] <= snapped[1:])  # Should be non-decreasing
        
    def test_snap_to_tolerance_multiple_groups(self):
        """Test snap_to_tolerance with multiple groups of close values."""
        knots = np.array([
            0.0, 0.0000001, 0.0000002,  # Group 1: should snap to 0.0
            1.0, 1.0000001, 1.0000002,  # Group 2: should snap to 1.0  
            2.0, 2.0000001, 2.0000002   # Group 3: should snap to 2.0
        ], dtype=np.float32)
        
        snapped = KnotsVector.snap_to_tolerance(knots)
        
        # Check that each group has been snapped to the same value
        assert snapped[0] == snapped[1] == snapped[2]  # First group
        assert snapped[3] == snapped[4] == snapped[5]  # Second group
        assert snapped[6] == snapped[7] == snapped[8]  # Third group
        
        # Check that different groups have different values
        assert snapped[0] != snapped[3]  # Group 1 != Group 2
        assert snapped[3] != snapped[6]  # Group 2 != Group 3
        
    def test_snap_to_tolerance_no_modification_needed(self):
        """Test snap_to_tolerance when no snapping is needed."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        snapped = KnotsVector.snap_to_tolerance(knots)
        
        # Should be identical since no values are close enough to snap
        np.testing.assert_array_equal(snapped, knots)
        
    def test_snap_to_tolerance_empty_array(self):
        """Test snap_to_tolerance with empty array."""
        knots = np.array([], dtype=np.float64)
        snapped = KnotsVector.snap_to_tolerance(knots)
        
        assert len(snapped) == 0
        assert snapped.dtype == knots.dtype
        
    def test_snap_to_tolerance_single_value(self):
        """Test snap_to_tolerance with single value."""
        knots = np.array([1.0], dtype=np.float64)
        snapped = KnotsVector.snap_to_tolerance(knots)
        
        np.testing.assert_array_equal(snapped, knots)
        
    def test_snap_to_tolerance_dtype_preservation(self):
        """Test that snap_to_tolerance preserves dtype."""
        for dtype in [np.float32, np.float64]:
            knots = np.array([0.0, 0.0000001, 1.0], dtype=dtype)
            snapped = KnotsVector.snap_to_tolerance(knots)
            
            assert snapped.dtype == dtype
            
    def test_snap_to_tolerance_does_not_modify_input(self):
        """Test that snap_to_tolerance doesn't modify the input array."""
        knots = np.array([0.0, 0.0000001, 1.0, 1.0000001], dtype=np.float32)
        original_knots = knots.copy()
        
        KnotsVector.snap_to_tolerance(knots)
        
        # Original array should be unchanged
        np.testing.assert_array_equal(knots, original_knots)


class TestCreateOpenUniformKnotVector:
    """Test create_open_uniform_knot_vector function."""

    def test_basic_creation(self):
        """Test basic open uniform knot vector creation."""
        kv = create_open_uniform_knot_vector(degree=2, start=0.0, end=1.0, num_intervals=2)
        
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
            kv = create_open_uniform_knot_vector(degree=degree, start=0.0, end=1.0, num_intervals=1)
            
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
            degree=1, start=0.0, end=1.0, num_intervals=1, dtype=np.float32
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
            create_open_uniform_knot_vector(degree=-1, start=0.0, end=1.0, num_intervals=1)

    def test_invalid_range(self):
        """Test creation with invalid range."""
        with pytest.raises(ValueError, match="Start must be less than end"):
            create_open_uniform_knot_vector(degree=1, start=1.0, end=0.0, num_intervals=1)
            
        with pytest.raises(ValueError, match="Start must be less than end"):
            create_open_uniform_knot_vector(degree=1, start=1.0, end=1.0, num_intervals=1)

    def test_invalid_num_intervals(self):
        """Test creation with invalid number of intervals."""
        with pytest.raises(ValueError, match="at least 1"):
            create_open_uniform_knot_vector(degree=1, start=0.0, end=1.0, num_intervals=0)

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
        kv = create_open_uniform_knot_vector(degree=2, start=0.0, end=1.0, num_intervals=4)
        
        # Should have 4 intervals: [0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]
        assert kv.num_nonzero_spans() == 4
        
        # Check uniform spacing
        assert kv.is_uniform()
        
        # Unique knots should be [0, 0.25, 0.5, 0.75, 1.0]
        expected_unique = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(kv.unique_knots, expected_unique)
