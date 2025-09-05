"""Test package imports and basic functionality."""

import numpy as np


def test_package_import():
    """Test that the package can be imported."""
    import dolfinx_iga

    assert hasattr(dolfinx_iga, "__version__")
