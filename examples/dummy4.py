import matplotlib.pyplot as plt
import numpy as np
from basix import CellType, ElementFamily, LagrangeVariant
from basix import create_element as create_element_basix

from dolfinx_iga.splines.basis_1D import evaluate_Bernstein_basis
from dolfinx_iga.splines.change_basis_1D import (
    create_Bernstein_to_Lagrange_basis_operator,
    create_Lagrange_to_Bernstein_basis_operator,
)

degree = 2
variant = LagrangeVariant.gll_centroid
C = create_Bernstein_to_Lagrange_basis_operator(degree, variant)

element = create_element_basix(ElementFamily.P, CellType.interval, degree, variant)


n_samples = 1000
tt = np.linspace(0, 1, n_samples, dtype=np.float64)
bernstein = evaluate_Bernstein_basis(degree, tt)
lagrange = bernstein @ C.T


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
fig.suptitle("From Bernstein to Lagrange")

for k in range(degree + 1):
    ax1.plot(tt, bernstein[:, k], label=f"B_{k}")
ax1.set_ylabel("Bernstein basis")
ax1.set_title(f"Bernstein basis polynomials (degree={degree})")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, lagrange[:, k], label=f"L_{k}")
ax2.set_xlabel("t")
ax2.set_ylabel("Lagrange basis")
ax2.set_title(f"Lagrange basis polynomials (degree={degree}, variant={variant.name})")
ax2.legend()

plt.tight_layout()


Cinv = create_Lagrange_to_Bernstein_basis_operator(degree, variant)
lagrange2 = element.tabulate(0, tt.reshape(-1, 1))[0, :, :, 0]
bernstein2 = lagrange2 @ Cinv.T

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
fig.suptitle("From Lagrange to Bernstein")

for k in range(degree + 1):
    ax1.plot(tt, bernstein2[:, k], label=f"B_{k}")
ax1.set_ylabel("Bernstein basis")
ax1.set_title(f"Bernstein basis polynomials (degree={degree})")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, lagrange2[:, k], label=f"L_{k}")
ax2.set_xlabel("t")
ax2.set_ylabel("Lagrange basis")
ax2.set_title(f"Lagrange basis polynomials (degree={degree}, variant={variant.name})")
ax2.legend()

plt.tight_layout()


plt.show()
# # points, weights = basix.make_quadrature(
# #     CellType.interval, 4, QuadratureType.gauss_jacobi
# # )
# # print(points)
# # print(weights)
# lagrange = basix.create_element(
#     ElementFamily.P, CellType.interval, 5, LagrangeVariant.gll_centroid
# )
# print(lagrange.points)
# lattice_points = basix.create_lattice(CellType.interval, 5, basix.LatticeType.gll, True)
# print(lattice_points)
