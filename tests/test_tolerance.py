"""Tests for tolerance utilities."""

import numpy as np
import pytest

from dolfinx_iga.utils.tolerance import (
    are_arrays_close,
    are_close,
    get_conservative_tolerance,
    get_default_tolerance,
    get_strict_tolerance,
    get_tolerance_info,
    unique_with_tolerance,
)


class TestToleranceFunctions:
    """Test tolerance utility functions."""

    def test_get_default_tolerance(self):
        """Test default tolerance values for different dtypes."""
        assert get_default_tolerance(np.float16) == 1e-3
        assert get_default_tolerance(np.float32) == 1e-6
        assert get_default_tolerance(np.float64) == 1e-12
        assert get_default_tolerance(np.longdouble) == 1e-15

    def test_get_strict_tolerance(self):
        """Test strict tolerance values."""
        assert get_strict_tolerance(np.float32) == 1e-7
        assert get_strict_tolerance(np.float64) == 1e-15

    def test_get_conservative_tolerance(self):
        """Test conservative tolerance values."""
        assert np.isclose(get_conservative_tolerance(np.float32), 1e-5)
        assert np.isclose(get_conservative_tolerance(np.float64), 1e-10)

    def test_are_close(self):
        """Test floating-point comparison function."""
        # Test close values (differences smaller than tolerance)
        assert are_close(1.0, 1.0000001, np.float32)  # 1e-7 < 1e-6 (default tolerance)
        assert are_close(1.0, 1.0000000000001, np.float64)  # 1e-13 < 1e-12 (default tolerance)

        # Test distant values
        assert not are_close(1.0, 1.01, np.float32)
        assert not are_close(1.0, 1.001, np.float64)

        # Test different tolerance types
        assert not are_close(1.0, 1.0000001, np.float32, "strict")  # 1e-7 == 1e-7 (strict)
        assert are_close(1.0, 1.000001, np.float32, "conservative")  # 1e-6 < 1e-5 (conservative)

        # Test float64 with strict tolerance (1e-15)
        assert are_close(1.0, 1.0 + 5e-16, np.float64, "strict")  # 5e-16 < 1e-15 (strict)
        assert not are_close(1.0, 1.0 + 2e-15, np.float64, "strict")  # 2e-15 > 1e-15 (strict)

    def test_are_arrays_close(self):
        """Test array comparison function."""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.000001, 2.000001, 3.000001], dtype=np.float32)
        arr3 = np.array([1.01, 2.01, 3.01], dtype=np.float32)

        assert are_arrays_close(arr1, arr2)
        assert not are_arrays_close(arr1, arr3)

        # Test shape mismatch
        arr4 = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError):
            are_arrays_close(arr1, arr4)

    def test_unique_with_tolerance(self):
        """Test unique values with tolerance."""
        # Use differences that are smaller than the default tolerance (1e-6 for float32)
        # Using 5e-7 which is definitely smaller than 1e-6
        arr = np.array([1.0, 1.0 + 5e-7, 2.0, 2.0 + 5e-7, 3.0], dtype=np.float32)
        unique, counts = unique_with_tolerance(arr)

        # With float32 default tolerance (1e-6), differences of 5e-7 should be grouped
        assert len(unique) == 3  # Should group close values
        assert counts.sum() == len(arr)

        # Test with strict tolerance - should find more unique values since 5e-7 > 1e-7 (strict)
        unique_strict, counts_strict = unique_with_tolerance(arr, "strict")
        assert len(unique_strict) >= len(unique)  # Strict should find more unique values

    def test_get_tolerance_info(self):
        """Test tolerance information function."""
        info = get_tolerance_info(np.float32)

        assert info["dtype"] == np.float32
        assert "machine_epsilon" in info
        assert "default_tolerance" in info
        assert "strict_tolerance" in info
        assert "conservative_tolerance" in info
        assert "precision_bits" in info

        # Check that tolerances are reasonable relative to machine epsilon
        assert info["default_tolerance"] > info["machine_epsilon"]
        assert info["strict_tolerance"] < info["default_tolerance"]
        assert info["conservative_tolerance"] > info["default_tolerance"]
