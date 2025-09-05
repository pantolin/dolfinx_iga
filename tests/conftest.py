"""Test configuration for dolfinx_iga."""

import numpy as np
import pytest


@pytest.fixture
def simple_control_points_2d():
    """Simple 2D control points for testing."""
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])


@pytest.fixture
def simple_control_points_3d():
    """Simple 3D control points for testing."""
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 1.0]]
    )
