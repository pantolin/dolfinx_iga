"""Tests for tolerance utilities."""

import numpy as np

from dolfinx_iga.utils.tolerance import (
    get_conservative_tolerance,
    get_default_tolerance,
    get_strict_tolerance,
    get_tolerance_info,
)


class TestToleranceFunctions:
    """Test tolerance utility functions."""

    def test_get_default_tolerance(self):
        """Test default tolerance values for different dtypes."""
        assert get_default_tolerance(np.float16) == np.float16(1e-3)
        assert get_default_tolerance(np.float32) == np.float32(1e-6)
        assert get_default_tolerance(np.float64) == np.float64(1e-12)
        assert get_default_tolerance(np.longdouble) == np.longdouble(1e-15)

    def test_get_strict_tolerance(self):
        """Test strict tolerance values."""
        assert get_strict_tolerance(np.float32) == np.float32(1e-7)
        assert get_strict_tolerance(np.float64) == np.float64(1e-15)

    def test_get_conservative_tolerance(self):
        """Test conservative tolerance values."""
        assert np.isclose(get_conservative_tolerance(np.float32), 1e-5)
        assert np.isclose(get_conservative_tolerance(np.float64), 1e-10)

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


if __name__ == "__main__":
    TestToleranceFunctions().test_get_default_tolerance()
