import matplotlib.pyplot as plt
import numpy as np

from dolfinx_iga.splines.basis_1D import (
    evaluate_Bernstein_basis,
    evaluate_Bspline_basis,
)
from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.change_basis_1D import create_Bezier_extraction_operators

# Example 1: Simple bspline curve.

# degree = 3
degree = 4
# num_spans = 10
# knots = create_uniform_open_knot_vector(num_spans, degree)
# knots = np.array(
#     [0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 7, 8, 9, 9, 9, 10, 10, 10, 10],
#     dtype=np.float64,
# )
# print(knots)
# # knots = np.array(
# #     [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.1, 8],
# #     dtype=np.float64,
# # )
knots = np.array(
    # [0, 1, 3, 6, 10, 11, 12, 13, 13, 13, 13],
    # [0, 1, 6, 6, 10, 11, 12, 13, 13, 13, 13],
    # [-1, 0, 6, 6, 6, 10, 11, 12, 13, 13, 13, 13, 13],
    # [-1, 0, 1, 6, 6, 10, 11, 12, 13, 13, 13, 13, 13],
    [-1, 0, 1, 3, 6, 10, 11, 12, 13, 13, 13, 13, 13],
    dtype=np.float64,
)
space = Bspline1D(knots, degree)
Cs = create_Bezier_extraction_operators(space)
# unique_knots = space.get_unique_knots()[0]
unique_knots = space.get_unique_knots_and_multiplicity(in_domain=True)[0]

n_samples = 100
tt = np.linspace(0.0, 0.99999, n_samples, dtype=np.float64)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i in range(space.get_num_intervals()):
    k0 = unique_knots[i]
    k1 = unique_knots[i + 1]
    bases = np.zeros((n_samples, degree + 1), dtype=np.float64)
    C = Cs[i]
    bases = evaluate_Bernstein_basis(degree, tt) @ C.T
    for k in range(degree + 1):
        ax1.plot(tt * (k1 - k0) + k0, bases[:, k])

    new_tt = tt * (k1 - k0) + k0
    bases_old = evaluate_Bspline_basis(space, new_tt)[0]
    for k in range(degree + 1):
        ax2.plot(tt * (k1 - k0) + k0, bases_old[:, k])

plt.show()
print("hola")
