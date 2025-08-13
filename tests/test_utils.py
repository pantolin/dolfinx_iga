"""Tests for utility functions."""

import numpy as np
import pytest
from dolfinx_iga.utils.knot_vector_utils import (
    generate_uniform_knot_vector,
    generate_open_knot_vector,
    generate_periodic_knot_vector,
    validate_knot_vector,
    normalize_knot_vector,
    find_knot_multiplicity,
)
from dolfinx_iga.utils.basis_functions import (
    find_span,
    bspline_basis,
    bspline_basis_derivatives,
)


class TestKnotVectorUtils:
    """Test knot vector utility functions."""
    
    def test_generate_uniform_knot_vector(self):
        """Test uniform knot vector generation."""
        n_control_points = 4
        degree = 2
        knots = generate_uniform_knot_vector(n_control_points, degree)
        
        expected_length = n_control_points + degree + 1
        assert len(knots) == expected_length
        
        # Check clamped ends
        assert np.all(knots[:degree + 1] == 0.0)
        assert np.all(knots[-(degree + 1):] == 1.0)
        
        # Check non-decreasing
        assert np.all(np.diff(knots) >= 0)
    
    def test_generate_open_knot_vector(self):
        """Test open knot vector generation."""
        n_control_points = 5
        degree = 3
        knots = generate_open_knot_vector(n_control_points, degree)
        
        expected_length = n_control_points + degree + 1
        assert len(knots) == expected_length
        
        # Should be same as uniform for now
        uniform_knots = generate_uniform_knot_vector(n_control_points, degree)
        np.testing.assert_array_equal(knots, uniform_knots)
    
    def test_generate_periodic_knot_vector(self):
        """Test periodic knot vector generation."""
        n_control_points = 6
        degree = 2
        knots = generate_periodic_knot_vector(n_control_points, degree)
        
        expected_length = n_control_points + degree + 1
        assert len(knots) == expected_length
        
        # Check that it's uniformly spaced
        spacing = np.diff(knots)
        np.testing.assert_allclose(spacing, spacing[0], atol=1e-12)
    
    def test_validate_knot_vector_valid(self):
        """Test validation of valid knot vector."""
        knots = np.array([0, 0, 0, 0.5, 1, 1, 1])
        n_control_points = 4
        degree = 2
        
        # Should not raise an exception
        validate_knot_vector(knots, n_control_points, degree)
    
    def test_validate_knot_vector_invalid_length(self):
        """Test validation fails for wrong length."""
        knots = np.array([0, 0, 0, 1, 1, 1])  # Too short
        n_control_points = 4
        degree = 2
        
        with pytest.raises(ValueError, match="length"):
            validate_knot_vector(knots, n_control_points, degree)
    
    def test_validate_knot_vector_non_decreasing(self):
        """Test validation fails for non-decreasing knot vector."""
        knots = np.array([0, 0, 1, 0.5, 1, 1, 1])  # Not non-decreasing
        n_control_points = 4
        degree = 2
        
        with pytest.raises(ValueError, match="non-decreasing"):
            validate_knot_vector(knots, n_control_points, degree)
    
    def test_normalize_knot_vector(self):
        """Test knot vector normalization."""
        knots = np.array([2, 2, 2, 3, 4, 4, 4])
        normalized = normalize_knot_vector(knots)
        
        assert normalized[0] == 0.0
        assert normalized[-1] == 1.0
        np.testing.assert_allclose(normalized, [0, 0, 0, 0.5, 1, 1, 1])
    
    def test_normalize_constant_knot_vector(self):
        """Test normalization fails for constant knot vector."""
        knots = np.array([1, 1, 1, 1])
        
        with pytest.raises(ValueError, match="constant"):
            normalize_knot_vector(knots)
    
    def test_find_knot_multiplicity(self):
        """Test finding knot multiplicity."""
        knots = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1])
        
        assert find_knot_multiplicity(knots, 0.0) == 3
        assert find_knot_multiplicity(knots, 0.5) == 2
        assert find_knot_multiplicity(knots, 1.0) == 3
        assert find_knot_multiplicity(knots, 0.25) == 0


class TestBasisFunctions:
    """Test B-spline basis function computations."""
    
    def test_find_span(self):
        """Test knot span finding."""
        knots = np.array([0, 0, 0, 0.5, 1, 1, 1])
        degree = 2
        
        assert find_span(0.0, degree, knots) == 2
        assert find_span(0.25, degree, knots) == 2
        assert find_span(0.5, degree, knots) == 3
        assert find_span(0.75, degree, knots) == 3
        assert find_span(1.0, degree, knots) == 3
    
    def test_bspline_basis_partition_of_unity(self):
        """Test that basis functions sum to 1."""
        knots = np.array([0, 0, 0, 0.5, 1, 1, 1])
        degree = 2
        n_basis = 4
        
        u_vals = np.linspace(0, 1, 20)
        for u in u_vals:
            basis = bspline_basis(u, degree, knots, n_basis)
            np.testing.assert_allclose(np.sum(basis), 1.0, atol=1e-12)
    
    def test_bspline_basis_non_negative(self):
        """Test that basis functions are non-negative."""
        knots = np.array([0, 0, 0, 0.5, 1, 1, 1])
        degree = 2
        n_basis = 4
        
        u_vals = np.linspace(0, 1, 20)
        for u in u_vals:
            basis = bspline_basis(u, degree, knots, n_basis)
            assert np.all(basis >= 0)
    
    def test_bspline_basis_derivatives_consistency(self):
        """Test that derivative computation is consistent with finite differences."""
        knots = np.array([0, 0, 0, 0.5, 1, 1, 1])
        degree = 2
        n_basis = 4
        u = 0.3
        h = 1e-8
        
        # Compute analytical derivative
        derivs = bspline_basis_derivatives(u, degree, knots, n_basis, 1)
        analytical_deriv = derivs[1, :]
        
        # Compute finite difference
        basis_plus = bspline_basis(u + h, degree, knots, n_basis)
        basis_minus = bspline_basis(u - h, degree, knots, n_basis)
        finite_diff = (basis_plus - basis_minus) / (2 * h)
        
        np.testing.assert_allclose(analytical_deriv, finite_diff, atol=1e-6)
    
    def test_bspline_basis_local_support(self):
        """Test that basis functions have local support."""
        knots = np.array([0, 0, 0, 0.5, 1, 1, 1])
        degree = 2
        n_basis = 4
        
        # At u=0.25, only certain basis functions should be non-zero
        basis = bspline_basis(0.25, degree, knots, n_basis)
        non_zero_indices = np.where(basis > 1e-12)[0]
        
        # For degree 2, at most 3 basis functions should be non-zero
        assert len(non_zero_indices) <= degree + 1
