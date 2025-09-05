#!/usr/bin/env python3
"""Simple test for the Cox-de Boor basis function evaluation."""

import numpy as np

from dolfinx_iga.bspline_1D import Bspline1D


def test_cox_de_boor_basic():
    """Test basic Cox-de Boor evaluation."""
    # Create a simple B-spline with open ends
    degree = 2
    knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64)

    bspline = Bspline1D(knots, degree)

    # Test evaluation at parameter t = 1.5
    t = np.float64(1.5)
    basis_values, first_index = bspline.evaluate_basis_functions(t)

    print(f"Parameter t = {t}")
    print(f"First non-zero basis index: {first_index}")
    print(f"Basis values: {basis_values}")
    print(f"Sum of basis values: {np.sum(basis_values)}")

    # Basis functions should sum to 1 (partition of unity property)
    assert abs(np.sum(basis_values) - 1.0) < 1e-12, "Basis functions should sum to 1"

    # Test at domain boundaries
    t_start = np.float64(0.0)
    basis_start, idx_start = bspline.evaluate_basis_functions(t_start)
    print(f"\nAt t = {t_start}: basis = {basis_start}, first_idx = {idx_start}")

    t_end = np.float64(3.0)
    basis_end, idx_end = bspline.evaluate_basis_functions(t_end)
    print(f"At t = {t_end}: basis = {basis_end}, first_idx = {idx_end}")

    # Test that we get the expected number of non-zero basis functions
    assert len(basis_values) == degree + 1, f"Should have {degree + 1} basis functions"

    print("\nTest passed!")


if __name__ == "__main__":
    test_cox_de_boor_basic()
