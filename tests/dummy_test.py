"""1D Poisson problem with DOLFINx: -u'' = f on [0,1], u(0)=u(1)=0"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


def solve_poisson_1d():
    """Solve -u'' = f with u(0) = u(1) = 0, f = 8π²sin(2πx)

    Exact solution: u = sin(2πx)
    """
    # Create 1D mesh on [0, 1]
    n_elem = 32
    domain = mesh.create_interval(MPI.COMM_WORLD, n_elem, [0.0, 1.0])

    # Define function space (degree 2)
    V = fem.functionspace(domain, ("Lagrange", 2, 2))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Source term: f = 8π²sin(2πx)
    x = ufl.SpatialCoordinate(domain)
    f = 4 * np.pi**2 * ufl.sin(2 * np.pi * x[0])

    # Weak form: a(u,v) = L(v)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # Dirichlet BCs: u(0) = u(1) = 0
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0

    def boundary(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))

    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    problem = LinearProblem(a, L, bcs=[bc])
    uh = problem.solve()

    # Compute L2 error vs exact solution u_ex = sin(2πx)
    x_vals = V.tabulate_dof_coordinates()[:, 0]
    u_exact = np.sin(2 * np.pi * x_vals)
    error_l2 = np.sqrt(np.sum((uh.x.array - u_exact) ** 2) / len(x_vals))

    return uh, error_l2, domain


def test_poisson_convergence():
    """Verify L2 error is small for refined mesh"""
    uh, error_l2, domain = solve_poisson_1d()
    print(f"L2 error: {error_l2:.6e}")
    assert error_l2 < 1e-3, f"Error too large: {error_l2}"


if __name__ == "__main__":
    uh, error_l2, domain = solve_poisson_1d()
    print("Solved 1D Poisson: -u'' = 8π²sin(2πx)")
    print(f"L2 error vs exact: {error_l2:.6e}")

    # Optional: plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        V = uh.function_space
        x_vals = V.tabulate_dof_coordinates()[:, 0]
        idx = np.argsort(x_vals)

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals[idx], uh.x.array[idx], "o-", label="Numerical")
        x_fine = np.linspace(0, 1, 200)
        plt.plot(x_fine, np.sin(2 * np.pi * x_fine), "--", label="Exact")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("1D Poisson: $-u''=8π²\\sin(2πx)$")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
