"""Tests for basis module (multi-dimensional basis functions)."""

import numpy as np
import pytest
from basix import LagrangeVariant

from dolfinx_iga.splines.basis import (
    evaluate_Bernstein_basis,
    evaluate_cardinal_Bspline_basis,
    evaluate_Lagrange_basis,
)


class TestEvaluateBernsteinBasis:
    """Test multi-dimensional Bernstein basis function evaluation."""

    def test_1d_scalar_input(self):
        """Test 1D Bernstein basis with scalar input."""
        degrees = [2]
        result = evaluate_Bernstein_basis(degrees, 0.5)
        expected = np.array([[0.25, 0.5, 0.25]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_array_input(self):
        """Test 1D Bernstein basis evaluation with array input."""
        degrees = [2]
        pts = np.array([0.0, 0.5, 1.0])
        result = evaluate_Bernstein_basis(degrees, pts)
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_tensor_grid_list(self):
        """Test 2D Bernstein basis on tensor grid (list format)."""
        degrees = [1, 1]
        pts = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        result = evaluate_Bernstein_basis(degrees, pts)
        # 4 basis functions, 4 points
        assert result.shape == (4, 4)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(4))

    def test_2d_scattered_points(self):
        """Test 2D Bernstein basis on scattered points."""
        degrees = [1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = evaluate_Bernstein_basis(degrees, pts)
        assert result.shape == (3, 4)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(3))

    def test_3d_evaluation(self):
        """Test 3D Bernstein basis evaluation."""
        degrees = [1, 1, 1]
        pts = np.array([[0.5, 0.5, 0.5]])
        result = evaluate_Bernstein_basis(degrees, pts)
        # 8 basis functions (2^3), 1 point
        assert result.shape == (1, 8)
        # Check partition of unity
        assert np.isclose(np.sum(result), 1.0)

    def test_partition_of_unity_random_points(self):
        """Test Bernstein basis partition of unity property with random points."""
        np.random.seed(42)
        for dim in [1, 2, 3]:
            degrees = [2] * dim
            pts = np.random.rand(10, dim)
            result = evaluate_Bernstein_basis(degrees, pts)
            sums = np.sum(result, axis=1)
            np.testing.assert_array_almost_equal(sums, np.ones(10))

    def test_different_degrees_per_dimension(self):
        """Test Bernstein basis with different degrees per dimension."""
        degrees = [1, 2]
        pts = np.array([[0.5, 0.5]])
        result = evaluate_Bernstein_basis(degrees, pts)
        # 2 * 3 = 6 basis functions
        assert result.shape == (1, 6)
        assert np.isclose(np.sum(result), 1.0)

    def test_c_order_functions(self):
        """Test C-order for functions."""
        degrees = [1, 1]
        pts = np.array([[0.5, 0.5]])
        result_f = evaluate_Bernstein_basis(degrees, pts, funcs_order="F")
        result_c = evaluate_Bernstein_basis(degrees, pts, funcs_order="C")
        # Results should be different due to ordering
        assert result_f.shape == result_c.shape
        # But same values in different order
        np.testing.assert_array_almost_equal(np.sort(result_f[0]), np.sort(result_c[0]))

    def test_tensor_grid_c_order_points(self):
        """Test tensor grid with C-order points."""
        degrees = [2, 2]
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        result = evaluate_Bernstein_basis(degrees, [x, y], pts_order="C")
        assert result.shape == (9, 9)
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(9))

    def test_tensor_grid_f_order_points(self):
        """Test tensor grid with F-order points."""
        degrees = [2, 2]
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        result = evaluate_Bernstein_basis(degrees, [x, y], pts_order="F")
        assert result.shape == (9, 9)
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(9))

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="All degrees must be non-negative"):
            evaluate_Bernstein_basis([1, -1], [0.5, 0.5])

    def test_invalid_funcs_order(self):
        """Test invalid functions order raises error."""
        with pytest.raises(ValueError, match="Invalid functions order"):
            evaluate_Bernstein_basis([1], [0.5], funcs_order="X")

    def test_invalid_pts_order(self):
        """Test invalid points order raises error."""
        with pytest.raises(ValueError, match="Invalid points order"):
            evaluate_Bernstein_basis([1], [0.5], pts_order="X")


class TestEvaluateCardinalBsplineBasis:
    """Test multi-dimensional cardinal B-spline basis function evaluation."""

    def test_1d_scalar_input(self):
        """Test 1D cardinal B-spline with scalar input."""
        degrees = [2]
        result = evaluate_cardinal_Bspline_basis(degrees, 0.5)
        # Check shape and partition of unity
        assert result.shape == (1, 3)
        assert np.isclose(np.sum(result), 1.0)

    def test_1d_array_input(self):
        """Test 1D cardinal B-spline basis evaluation."""
        degrees = [2]
        pts = np.array([0.0, 0.5, 1.0])
        result = evaluate_cardinal_Bspline_basis(degrees, pts)
        # Check shape
        assert result.shape == (3, 3)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(3))

    def test_2d_tensor_grid(self):
        """Test 2D cardinal B-spline basis evaluation."""
        degrees = [2, 2]
        pts = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
        result = evaluate_cardinal_Bspline_basis(degrees, pts)
        # 9 basis functions (3x3), 9 points
        assert result.shape == (9, 9)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(9))

    def test_2d_scattered_points(self):
        """Test 2D cardinal B-spline on scattered points."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = evaluate_cardinal_Bspline_basis(degrees, pts)
        assert result.shape == (3, 9)
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(3))

    def test_3d_evaluation(self):
        """Test 3D cardinal B-spline evaluation."""
        degrees = [1, 1, 1]
        pts = np.array([[0.5, 0.5, 0.5]])
        result = evaluate_cardinal_Bspline_basis(degrees, pts)
        # 8 basis functions (2^3), 1 point
        assert result.shape == (1, 8)
        assert np.isclose(np.sum(result), 1.0)

    def test_partition_of_unity_random_points(self):
        """Test cardinal B-spline partition of unity property."""
        np.random.seed(42)
        for dim in [1, 2, 3]:
            degrees = [2] * dim
            pts = np.random.rand(10, dim)
            result = evaluate_cardinal_Bspline_basis(degrees, pts)
            sums = np.sum(result, axis=1)
            np.testing.assert_array_almost_equal(sums, np.ones(10))

    def test_different_degrees_per_dimension(self):
        """Test cardinal B-spline with different degrees per dimension."""
        degrees = [1, 3]
        pts = np.array([[0.5, 0.5]])
        result = evaluate_cardinal_Bspline_basis(degrees, pts)
        # 2 * 4 = 8 basis functions
        assert result.shape == (1, 8)
        assert np.isclose(np.sum(result), 1.0)

    def test_high_degree(self):
        """Test cardinal B-spline with high degree."""
        degrees = [5]
        pts = np.linspace(0.0, 1.0, 11)
        result = evaluate_cardinal_Bspline_basis(degrees, pts)
        assert result.shape == (11, 6)
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(11))

    def test_negative_degree_error(self):
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="All degrees must be non-negative"):
            evaluate_cardinal_Bspline_basis([2, -1], [0.5, 0.5])


class TestEvaluateLagrangeBasis:
    """Test multi-dimensional Lagrange basis function evaluation."""

    def test_1d_scalar_input(self):
        """Test 1D Lagrange with scalar input."""
        degrees = [2]
        result = evaluate_Lagrange_basis(degrees, 0.5)
        # Check shape and partition of unity
        assert result.shape == (1, 3)
        assert np.isclose(np.sum(result), 1.0)

    def test_1d_array_input(self):
        """Test 1D Lagrange basis evaluation."""
        degrees = [2]
        pts = np.array([0.0, 0.5, 1.0])
        result = evaluate_Lagrange_basis(degrees, pts)
        # Check shape: 3 points, 3 basis functions
        assert result.shape == (3, 3)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(3))

    def test_1d_evaluation_at_nodes(self):
        """Test 1D Lagrange basis at nodal points."""
        degrees = [1]
        pts = np.array([0.0, 1.0])
        result = evaluate_Lagrange_basis(degrees, pts)
        # At nodes, basis should be identity-like
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [0.0, 1.0])

    def test_2d_tensor_grid(self):
        """Test 2D Lagrange basis on tensor grid."""
        degrees = [2, 2]
        pts = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
        result = evaluate_Lagrange_basis(degrees, pts)
        # 9 basis functions (3x3), 9 points
        assert result.shape == (9, 9)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(9))

    def test_2d_scattered_points(self):
        """Test 2D Lagrange basis on scattered points."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = evaluate_Lagrange_basis(degrees, pts)
        assert result.shape == (3, 9)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(3))

    def test_3d_evaluation(self):
        """Test 3D Lagrange evaluation."""
        degrees = [1, 1, 1]
        pts = np.array([[0.5, 0.5, 0.5]])
        result = evaluate_Lagrange_basis(degrees, pts)
        # 8 basis functions (2^3), 1 point
        assert result.shape == (1, 8)
        assert np.isclose(np.sum(result), 1.0)

    def test_partition_of_unity_random_points(self):
        """Test Lagrange basis partition of unity property."""
        np.random.seed(42)
        for dim in [1, 2]:
            degrees = [2] * dim
            pts = np.random.rand(10, dim)
            result = evaluate_Lagrange_basis(degrees, pts)
            sums = np.sum(result, axis=1)
            np.testing.assert_array_almost_equal(sums, np.ones(10))

    def test_different_degrees_per_dimension(self):
        """Test Lagrange with different degrees per dimension."""
        degrees = [1, 3]
        pts = np.array([[0.5, 0.5]])
        result = evaluate_Lagrange_basis(degrees, pts)
        # 2 * 4 = 8 basis functions
        assert result.shape == (1, 8)
        assert np.isclose(np.sum(result), 1.0)

    def test_high_degree(self):
        """Test Lagrange with high degree."""
        degrees = [5]
        pts = np.linspace(0.0, 1.0, 11)
        result = evaluate_Lagrange_basis(degrees, pts)
        assert result.shape == (11, 6)
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(11))

    def test_gll_variant(self):
        """Test Lagrange basis with GLL variant."""
        degrees = [3]
        pts = np.linspace(0.0, 1.0, 10)
        result = evaluate_Lagrange_basis(
            degrees, pts, lagrange_variant=LagrangeVariant.gll_warped
        )
        # Check shape: 10 points, 4 basis functions (degree+1)
        assert result.shape == (10, 4)
        # Check partition of unity
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(10))

    def test_degree_zero_error(self):
        """Test that degree 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="All Lagrange basis degrees must be at least 1"
        ):
            evaluate_Lagrange_basis([0], [0.5])

    def test_invalid_funcs_order(self):
        """Test invalid functions order raises error."""
        with pytest.raises(ValueError, match="Invalid functions order"):
            evaluate_Lagrange_basis([1], [0.5], funcs_order="X")


class TestFunctionOrdering:
    """Test that function ordering (F vs C) is correct."""

    def test_bernstein_2d_f_order_functions(self):
        """Test 2D Bernstein with F-order functions using nested loops."""
        degrees = [2, 1]  # 3 basis functions in x, 2 in y
        pts = np.array([[0.3, 0.7]])

        # Compute using library with F-order (default)
        result = evaluate_Bernstein_basis(degrees, pts, funcs_order="F")

        # Compute reference using explicit nested loops
        # F-order: first index varies fastest, so loop order is (j, i) for indices (i, j)
        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        vals_x = evaluate_Bernstein_basis_1D(degrees[0], pts[:, 0])  # shape (1, 3)
        vals_y = evaluate_Bernstein_basis_1D(degrees[1], pts[:, 1])  # shape (1, 2)

        reference = np.zeros((1, 6))  # 3*2 = 6 basis functions
        idx = 0
        for j in range(degrees[1] + 1):  # y direction (second index)
            for i in range(degrees[0] + 1):  # x direction (first index, varies fastest)
                reference[0, idx] = vals_x[0, i] * vals_y[0, j]
                idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_bernstein_2d_c_order_functions(self):
        """Test 2D Bernstein with C-order functions using nested loops."""
        degrees = [2, 1]  # 3 basis functions in x, 2 in y
        pts = np.array([[0.3, 0.7]])

        # Compute using library with C-order
        result = evaluate_Bernstein_basis(degrees, pts, funcs_order="C")

        # Compute reference using explicit nested loops
        # C-order: last index varies fastest, so loop order is (i, j) for indices (i, j)
        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        vals_x = evaluate_Bernstein_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_Bernstein_basis_1D(degrees[1], pts[:, 1])

        reference = np.zeros((1, 6))
        idx = 0
        for i in range(degrees[0] + 1):  # x direction (first index)
            for j in range(degrees[1] + 1):  # y direction (last index, varies fastest)
                reference[0, idx] = vals_x[0, i] * vals_y[0, j]
                idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_bspline_2d_f_order_functions(self):
        """Test 2D cardinal B-spline with F-order functions using nested loops."""
        degrees = [2, 1]
        pts = np.array([[0.4, 0.6]])

        result = evaluate_cardinal_Bspline_basis(degrees, pts, funcs_order="F")

        from dolfinx_iga.splines.basis_1D import evaluate_cardinal_Bspline_basis_1D

        vals_x = evaluate_cardinal_Bspline_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_cardinal_Bspline_basis_1D(degrees[1], pts[:, 1])

        reference = np.zeros((1, 6))
        idx = 0
        for j in range(degrees[1] + 1):
            for i in range(degrees[0] + 1):
                reference[0, idx] = vals_x[0, i] * vals_y[0, j]
                idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_bspline_2d_c_order_functions(self):
        """Test 2D cardinal B-spline with C-order functions using nested loops."""
        degrees = [2, 1]
        pts = np.array([[0.4, 0.6]])

        result = evaluate_cardinal_Bspline_basis(degrees, pts, funcs_order="C")

        from dolfinx_iga.splines.basis_1D import evaluate_cardinal_Bspline_basis_1D

        vals_x = evaluate_cardinal_Bspline_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_cardinal_Bspline_basis_1D(degrees[1], pts[:, 1])

        reference = np.zeros((1, 6))
        idx = 0
        for i in range(degrees[0] + 1):
            for j in range(degrees[1] + 1):
                reference[0, idx] = vals_x[0, i] * vals_y[0, j]
                idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_lagrange_2d_f_order_functions(self):
        """Test 2D Lagrange with F-order functions using nested loops."""
        degrees = [2, 1]
        pts = np.array([[0.35, 0.65]])

        result = evaluate_Lagrange_basis(degrees, pts, funcs_order="F")

        from dolfinx_iga.splines.basis_1D import evaluate_Lagrange_basis_1D

        vals_x = evaluate_Lagrange_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_Lagrange_basis_1D(degrees[1], pts[:, 1])

        reference = np.zeros((1, 6))
        idx = 0
        for j in range(degrees[1] + 1):
            for i in range(degrees[0] + 1):
                reference[0, idx] = vals_x[0, i] * vals_y[0, j]
                idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_lagrange_2d_c_order_functions(self):
        """Test 2D Lagrange with C-order functions using nested loops."""
        degrees = [2, 1]
        pts = np.array([[0.35, 0.65]])

        result = evaluate_Lagrange_basis(degrees, pts, funcs_order="C")

        from dolfinx_iga.splines.basis_1D import evaluate_Lagrange_basis_1D

        vals_x = evaluate_Lagrange_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_Lagrange_basis_1D(degrees[1], pts[:, 1])

        reference = np.zeros((1, 6))
        idx = 0
        for i in range(degrees[0] + 1):
            for j in range(degrees[1] + 1):
                reference[0, idx] = vals_x[0, i] * vals_y[0, j]
                idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_3d_f_order_functions(self):
        """Test 3D Bernstein with F-order functions using nested loops."""
        degrees = [1, 2, 1]  # 2, 3, 2 basis functions
        pts = np.array([[0.3, 0.5, 0.7]])

        result = evaluate_Bernstein_basis(degrees, pts, funcs_order="F")

        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        vals_x = evaluate_Bernstein_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_Bernstein_basis_1D(degrees[1], pts[:, 1])
        vals_z = evaluate_Bernstein_basis_1D(degrees[2], pts[:, 2])

        reference = np.zeros((1, 12))  # 2*3*2 = 12
        idx = 0
        for k in range(degrees[2] + 1):  # z (third index)
            for j in range(degrees[1] + 1):  # y (second index)
                for i in range(degrees[0] + 1):  # x (first index, varies fastest)
                    reference[0, idx] = vals_x[0, i] * vals_y[0, j] * vals_z[0, k]
                    idx += 1

        np.testing.assert_array_almost_equal(result, reference)

    def test_3d_c_order_functions(self):
        """Test 3D Bernstein with C-order functions using nested loops."""
        degrees = [1, 2, 1]
        pts = np.array([[0.3, 0.5, 0.7]])

        result = evaluate_Bernstein_basis(degrees, pts, funcs_order="C")

        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        vals_x = evaluate_Bernstein_basis_1D(degrees[0], pts[:, 0])
        vals_y = evaluate_Bernstein_basis_1D(degrees[1], pts[:, 1])
        vals_z = evaluate_Bernstein_basis_1D(degrees[2], pts[:, 2])

        reference = np.zeros((1, 12))
        idx = 0
        for i in range(degrees[0] + 1):  # x (first index)
            for j in range(degrees[1] + 1):  # y (second index)
                for k in range(degrees[2] + 1):  # z (last index, varies fastest)
                    reference[0, idx] = vals_x[0, i] * vals_y[0, j] * vals_z[0, k]
                    idx += 1

        np.testing.assert_array_almost_equal(result, reference)


class TestPointOrdering:
    """Test that point ordering (F vs C) is correct for tensor grids."""

    def test_bernstein_2d_f_order_points(self):
        """Test 2D Bernstein with F-order points using nested loops."""
        degrees = [2, 1]
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 1.0])

        # Compute using library with F-order points
        result = evaluate_Bernstein_basis(degrees, [x, y], pts_order="F")

        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        # Compute reference: F-order means first index varies fastest
        # Points ordered as: (x[0],y[0]), (x[1],y[0]), (x[2],y[0]), (x[0],y[1]), ...
        # Functions ordered in F-order: first index varies fastest
        reference = []
        for j in range(len(y)):  # y (second index)
            for i in range(len(x)):  # x (first index, varies fastest)
                vals_x = evaluate_Bernstein_basis_1D(degrees[0], np.array([x[i]]))
                vals_y = evaluate_Bernstein_basis_1D(degrees[1], np.array([y[j]]))
                # Compute all 6 basis functions at this point with F-order
                basis_vals = []
                for jj in range(degrees[1] + 1):
                    for ii in range(degrees[0] + 1):
                        basis_vals.append(vals_x[0, ii] * vals_y[0, jj])
                reference.append(basis_vals)

        reference = np.array(reference)
        np.testing.assert_array_almost_equal(result, reference)

    def test_bernstein_2d_c_order_points(self):
        """Test 2D Bernstein with C-order points using nested loops."""
        degrees = [2, 1]
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 1.0])

        # Compute using library with C-order points
        result = evaluate_Bernstein_basis(degrees, [x, y], pts_order="C")

        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        # Compute reference: C-order means last index varies fastest
        # Points ordered as: (x[0],y[0]), (x[0],y[1]), (x[1],y[0]), (x[1],y[1]), ...
        # Functions still ordered in F-order (default): first index varies fastest
        reference = []
        for i in range(len(x)):  # x (first index)
            for j in range(len(y)):  # y (last index, varies fastest)
                vals_x = evaluate_Bernstein_basis_1D(degrees[0], np.array([x[i]]))
                vals_y = evaluate_Bernstein_basis_1D(degrees[1], np.array([y[j]]))
                # Compute all 6 basis functions at this point (F-order for functions)
                basis_vals = []
                for jj in range(degrees[1] + 1):
                    for ii in range(degrees[0] + 1):
                        basis_vals.append(vals_x[0, ii] * vals_y[0, jj])
                reference.append(basis_vals)

        reference = np.array(reference)
        np.testing.assert_array_almost_equal(result, reference)

    def test_bspline_2d_tensor_grid_ordering(self):
        """Test 2D cardinal B-spline tensor grid point ordering."""
        degrees = [2, 1]
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.3, 0.7])

        # F-order (default)
        result_f = evaluate_cardinal_Bspline_basis(degrees, [x, y], pts_order="F")

        from dolfinx_iga.splines.basis_1D import evaluate_cardinal_Bspline_basis_1D

        reference_f = []
        for j in range(len(y)):
            for i in range(len(x)):
                vals_x = evaluate_cardinal_Bspline_basis_1D(
                    degrees[0], np.array([x[i]])
                )
                vals_y = evaluate_cardinal_Bspline_basis_1D(
                    degrees[1], np.array([y[j]])
                )
                # F-order for functions
                basis_vals = []
                for jj in range(degrees[1] + 1):
                    for ii in range(degrees[0] + 1):
                        basis_vals.append(vals_x[0, ii] * vals_y[0, jj])
                reference_f.append(basis_vals)

        reference_f = np.array(reference_f)
        np.testing.assert_array_almost_equal(result_f, reference_f)

    def test_lagrange_2d_tensor_grid_ordering(self):
        """Test 2D Lagrange tensor grid point ordering."""
        degrees = [2, 1]
        x = np.array([0.1, 0.5, 0.9])
        y = np.array([0.2, 0.8])

        result = evaluate_Lagrange_basis(degrees, [x, y], pts_order="F")

        from dolfinx_iga.splines.basis_1D import evaluate_Lagrange_basis_1D

        reference = []
        for j in range(len(y)):
            for i in range(len(x)):
                vals_x = evaluate_Lagrange_basis_1D(degrees[0], np.array([x[i]]))
                vals_y = evaluate_Lagrange_basis_1D(degrees[1], np.array([y[j]]))
                # F-order for functions
                basis_vals = []
                for jj in range(degrees[1] + 1):
                    for ii in range(degrees[0] + 1):
                        basis_vals.append(vals_x[0, ii] * vals_y[0, jj])
                reference.append(basis_vals)

        reference = np.array(reference)
        np.testing.assert_array_almost_equal(result, reference)


class TestCombinedOrdering:
    """Test combined function and point ordering."""

    def test_bernstein_f_funcs_c_points(self):
        """Test Bernstein with F-order functions and C-order points."""
        degrees = [2, 1]
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = evaluate_Bernstein_basis(
            degrees, [x, y], funcs_order="F", pts_order="C"
        )

        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        # C-order points: x varies first, then y
        # F-order functions: for each point, functions ordered with first index varying fastest
        reference = []
        for i in range(len(x)):
            for j in range(len(y)):
                vals_x = evaluate_Bernstein_basis_1D(degrees[0], np.array([x[i]]))
                vals_y = evaluate_Bernstein_basis_1D(degrees[1], np.array([y[j]]))
                # F-order functions
                basis_vals = []
                for jj in range(degrees[1] + 1):
                    for ii in range(degrees[0] + 1):
                        basis_vals.append(vals_x[0, ii] * vals_y[0, jj])
                reference.append(basis_vals)

        reference = np.array(reference)
        np.testing.assert_array_almost_equal(result, reference)

    def test_bernstein_c_funcs_f_points(self):
        """Test Bernstein with C-order functions and F-order points."""
        degrees = [2, 1]
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = evaluate_Bernstein_basis(
            degrees, [x, y], funcs_order="C", pts_order="F"
        )

        from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis_1D

        # F-order points: x varies fastest, then y
        # C-order functions: for each point, functions ordered with last index varying fastest
        reference = []
        for j in range(len(y)):
            for i in range(len(x)):
                vals_x = evaluate_Bernstein_basis_1D(degrees[0], np.array([x[i]]))
                vals_y = evaluate_Bernstein_basis_1D(degrees[1], np.array([y[j]]))
                # C-order functions
                basis_vals = []
                for ii in range(degrees[0] + 1):
                    for jj in range(degrees[1] + 1):
                        basis_vals.append(vals_x[0, ii] * vals_y[0, jj])
                reference.append(basis_vals)

        reference = np.array(reference)
        np.testing.assert_array_almost_equal(result, reference)


class TestPointsOrderingConsistency:
    """Test consistency across different point orderings and formats."""

    def test_bernstein_tensor_grid_vs_meshgrid_c_order(self):
        """Test Bernstein: tensor grid list vs meshgrid array with C-order."""
        degrees = [2, 2]
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)

        # List format (tensor grid)
        result_list = evaluate_Bernstein_basis(degrees, [x, y], pts_order="C")

        # Meshgrid format with C-order
        xx, yy = np.meshgrid(x, y, indexing="ij")
        pts_array = np.column_stack([xx.ravel(), yy.ravel()])
        result_array = evaluate_Bernstein_basis(degrees, pts_array)

        np.testing.assert_array_almost_equal(result_list, result_array)

    def test_bernstein_tensor_grid_vs_meshgrid_f_order(self):
        """Test Bernstein: tensor grid list vs meshgrid array with F-order."""
        degrees = [2, 2]
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)

        # List format (tensor grid) with F-order
        result_list = evaluate_Bernstein_basis(degrees, [x, y], pts_order="F")

        # Meshgrid format with C-order
        xx, yy = np.meshgrid(x, y, indexing="xy")
        pts_array = np.column_stack([xx.ravel(), yy.ravel()])
        result_array = evaluate_Bernstein_basis(degrees, pts_array)

        np.testing.assert_array_almost_equal(result_list, result_array)

    def test_bspline_tensor_grid_vs_meshgrid(self):
        """Test cardinal B-spline: tensor grid vs meshgrid consistency."""
        degrees = [2, 2]
        x = np.linspace(0, 1, 4)
        y = np.linspace(0, 1, 4)

        result_list = evaluate_cardinal_Bspline_basis(degrees, [x, y], pts_order="C")

        xx, yy = np.meshgrid(x, y, indexing="ij")
        pts_array = np.column_stack([xx.ravel(), yy.ravel()])
        result_array = evaluate_cardinal_Bspline_basis(degrees, pts_array)

        np.testing.assert_array_almost_equal(result_list, result_array)

    def test_lagrange_tensor_grid_vs_meshgrid(self):
        """Test Lagrange: tensor grid vs meshgrid consistency."""
        degrees = [2, 2]
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)

        result_list = evaluate_Lagrange_basis(degrees, [x, y], pts_order="C")

        xx, yy = np.meshgrid(x, y, indexing="ij")
        pts_array = np.column_stack([xx.ravel(), yy.ravel()])
        result_array = evaluate_Lagrange_basis(degrees, pts_array)

        np.testing.assert_array_almost_equal(result_list, result_array)


class TestCrossBasisConsistency:
    """Test consistency properties across different basis types."""

    def test_degree_zero_bernstein_vs_bspline(self):
        """Test degree 0 Bernstein and B-spline are identical."""
        degrees = [0]
        pts = np.linspace(0.0, 1.0, 10)
        result_bernstein = evaluate_Bernstein_basis(degrees, pts)
        result_bspline = evaluate_cardinal_Bspline_basis(degrees, pts)
        # Both should be constant 1
        np.testing.assert_array_almost_equal(result_bernstein, result_bspline)
        np.testing.assert_array_almost_equal(result_bernstein, np.ones((10, 1)))

    def test_2d_all_bases_partition_of_unity(self):
        """Test all basis types satisfy partition of unity in 2D."""
        degrees = [2, 2]
        np.random.seed(123)
        pts = np.random.rand(20, 2)

        for basis_func in [
            evaluate_Bernstein_basis,
            evaluate_cardinal_Bspline_basis,
            evaluate_Lagrange_basis,
        ]:
            result = basis_func(degrees, pts)
            sums = np.sum(result, axis=1)
            np.testing.assert_array_almost_equal(
                sums, np.ones(20), err_msg=f"Failed for {basis_func.__name__}"
            )
