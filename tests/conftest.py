"""Test configuration for dolfinx_iga."""

import pytest
import numpy as np


@pytest.fixture
def simple_control_points_2d():
    """Simple 2D control points for testing."""
    return np.array([
        [0.0, 0.0],
        [1.0, 1.0], 
        [2.0, 0.0],
        [3.0, 1.0]
    ])


@pytest.fixture
def simple_control_points_3d():
    """Simple 3D control points for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [3.0, 1.0, 1.0]
    ])


@pytest.fixture
def simple_weights():
    """Simple weights for NURBS testing."""
    return np.array([1.0, 2.0, 1.0, 1.0])


@pytest.fixture
def surface_control_points():
    """Control points for surface testing."""
    # 3x3 grid of control points
    return np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        [[1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [1.0, 2.0, 1.0]],
        [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
    ])


@pytest.fixture
def surface_weights():
    """Weights for surface testing."""
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
