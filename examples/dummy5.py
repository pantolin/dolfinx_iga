import matplotlib.pyplot as plt
import numpy as np

from dolfinx_iga.splines.basis_1D import (
    evaluate_Bspline_basis,
    evaluate_cardinal_Bspline_basis,
)
from dolfinx_iga.splines.bspline_1D import (
    Bspline1D,
)
from dolfinx_iga.splines.knots import create_cardinal_Bspline_knot_vector

# Testing cardinal B-spline basis.


degree = 6
n_samples = 1000
tt = np.linspace(0, 0.9999, n_samples, dtype=np.float64)
basis_vals = np.zeros((n_samples, degree + 1), dtype=np.float64)

basis_vals = evaluate_cardinal_Bspline_basis(degree, tt)


# Construct Bezier-like knot vector for degree p: [0, ..., 0, 1, ..., 1] (p+1 zeros, p+1 ones)
knots = create_cardinal_Bspline_knot_vector(1, degree, np.float64)
spline = Bspline1D(knots, degree)
basis_vals_spline, _ = evaluate_Bspline_basis(spline, tt)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for k in range(degree + 1):
    ax1.plot(tt, basis_vals[:, k], label=f"B_{{{degree},{k}}}(t)")
ax1.set_ylabel("Basis value")
ax1.set_title(f"Cardinal B-spline (Bernstein) basis, degree={degree}")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, basis_vals_spline[:, k], label=f"B_{{{degree},{k}}}(t) (Bspline1D)")
ax2.set_xlabel("t")
ax2.set_ylabel("Basis value")
ax2.set_title(f"Cardinal B-spline (Bernstein) basis via Bspline1D, degree={degree}")
ax2.legend()

plt.tight_layout()
plt.show()

# import numpy as np
# from scipy.special import eval_sh_legendre  # For the example


# def get_gauss_quadrature(n_points: int):
#     """
#     Computes Gauss-Legendre quadrature points and weights for the [0, 1] interval.

#     Args:
#         n_points: The number of quadrature points to generate.

#     Returns:
#         A tuple containing:
#         - points (np.ndarray): The quadrature points in the [0, 1] interval.
#         - weights (np.ndarray): The corresponding quadrature weights.
#     """
#     # Get standard Gauss-Legendre points and weights on [-1, 1]
#     points_ref, weights_ref = np.polynomial.legendre.leggauss(n_points)

#     # Affine transformation to map points from [-1, 1] to [0, 1]
#     # t = (x + 1) / 2
#     points = (points_ref + 1) / 2

#     # Adjust weights for the change of interval: dt = dx / 2
#     weights = weights_ref / 2

#     return points, weights


# def compute_change_of_basis_matrix(basis_A, basis_B, dim: int, n_quad_points: int = 20):
#     """
#     Computes the change of basis matrix M to transform coefficients from
#     an old basis (A) to a new basis (B).

#     The matrix M satisfies: coeffs_B = M @ coeffs_A.

#     Args:
#         basis_A (callable): A function `basis_A(i, x)` that evaluates the i-th
#                             function of the old basis at point(s) x.
#         basis_B (callable): A function `basis_B(j, x)` that evaluates the j-th
#                             function of the new basis at point(s) x.
#         dim (int): The dimension of the function space (number of basis functions).
#         n_quad_points (int): The number of Gauss points for numerical integration.

#     Returns:
#         np.ndarray: The (dim x dim) change of basis matrix M.
#     """
#     # 1. Get Gauss quadrature points and weights for the inner product on [0,
#     # 1]
#     points, weights = get_gauss_quadrature(n_quad_points)

#     # 2. Pre-evaluate all basis functions at all quadrature points for
#     # efficiency
#     eval_A = np.array([basis_A(i, points) for i in range(dim)])
#     eval_B = np.array([basis_B(j, points) for j in range(dim)])

#     # 3. Compute the Gram matrix G for the new basis B: G_kj = <b_k, b_j>
#     # The inner product <f, g> is approximated by sum(w_m * f(x_m) * g(x_m))
#     G = np.zeros((dim, dim))
#     for k in range(dim):
#         for j in range(dim):
#             integrand = eval_B[k, :] * eval_B[j, :]
#             G[k, j] = np.dot(integrand, weights)

#     # 4. Compute the mixed inner product matrix C: C_ki = <a_i, b_k>
#     C = np.zeros((dim, dim))
#     for k in range(dim):
#         for i in range(dim):
#             integrand = eval_A[i, :] * eval_B[k, :]
#             C[k, i] = np.dot(integrand, weights)

#     # 5. Solve the system C = G M for M, which means M = G^-1 C
#     G_inv = np.linalg.inv(G)
#     M = G_inv @ C

#     return M


# # --- Example Usage ---
# if __name__ == "__main__":
#     # Define two 3-dimensional bases on the interval [0, 1]
#     DIM = 3

#     # Old Basis A: Monomial basis {1, x, x^2}
#     def monomial_basis(i, x):
#         return x**i

#     # New Basis B: Shifted Legendre polynomials (which are orthogonal on [0,1])
#     def shifted_legendre_basis(j, x):
#         return eval_sh_legendre(j, x)

#     print("Computing the change of basis matrix from Monomials to Shifted Legendre...")

#     # Compute the matrix
#     M = compute_change_of_basis_matrix(
#         basis_A=monomial_basis,
#         basis_B=shifted_legendre_basis,
#         dim=DIM,
#         n_quad_points=20,  # More points for higher accuracy
#     )

#     print("\nChange of Basis Matrix M:\n", np.round(M, 8))

#     # --- Verification ---
#     print("\n--- Verifying the transformation ---")

#     # Define a function f(x) = 4x^2 - 3x + 2
#     # Its coefficients in the monomial basis (A) are [2, -3, 4]
#     coeffs_A = np.array([2, -3, 4])
#     print(f"Original coefficients in Basis A (Monomial): {coeffs_A}")

#     # Use the matrix M to find the coefficients in the new basis (B)
#     coeffs_B = M @ coeffs_A
#     print(f"Transformed coefficients in Basis B (Legendre): {coeffs_B}")

#     # Check: Does the function remain the same?
#     # Let's evaluate the function using both representations at some test
#     # points.
#     test_points = np.linspace(0, 1, 5)

#     # Evaluation using Basis A
#     y_from_A = sum(coeffs_A[i] * monomial_basis(i, test_points) for i in range(DIM))

#     # Evaluation using Basis B
#     y_from_B = sum(
#         coeffs_B[j] * shifted_legendre_basis(j, test_points) for j in range(DIM)
#     )

#     print("\nEvaluating the function at several points using both bases:")
#     print("From Basis A:", np.round(y_from_A, 8))
#     print("From Basis B:", np.round(y_from_B, 8))

#     # The two results should be identical
#     assert np.allclose(y_from_A, y_from_B), "Verification failed!"
#     print("\n✅ Verification successful! The representations are equivalent.")
