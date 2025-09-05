import matplotlib.pyplot as plt
import numpy as np

from dolfinx_iga.splines.basis_1D import (
    evaluate_monomial_basis,
)
from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.change_basis_1D import (
    create_cardinal_to_monomial_basis_operator,
    create_monomial_to_cardinal_basis_operator,
    evaluate_cardinal_Bspline_basis,
)
from dolfinx_iga.splines.curve import BsplineCurve
from dolfinx_iga.splines.knots import create_uniform_open_knot_vector

dtype = np.float64


knots = create_uniform_open_knot_vector(2, 2)
space = Bspline1D(knots, 2)
control_points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
curve = BsplineCurve(space, control_points)
print(curve.evaluate([0.5, 0.75, 0.85]))

degree = 4
C = create_monomial_to_cardinal_basis_operator(degree, dtype)


n_samples = 1000
tt = np.linspace(0, 1, n_samples, dtype=dtype)
monomial = evaluate_monomial_basis(degree, tt)
# cardinal = evaluate_cardinal_Bspline_basis(degree, tt)
cardinal = monomial @ C.T


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
fig.suptitle("From monomials to cardinal")

for k in range(degree + 1):
    ax1.plot(tt, monomial[:, k], label=f"t^{k}")
ax1.set_ylabel("Monomial basis")
ax1.set_title(f"Monomial basis polynomials (degree={degree})")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, cardinal[:, k], label=f"C_{k}")
ax2.set_xlabel("t")
ax2.set_ylabel("Cardinal basis")
ax2.set_title(f"Cardinal basis polynomials (degree={degree})")
ax2.legend()

plt.tight_layout()


Cinv = create_cardinal_to_monomial_basis_operator(degree, dtype)
cardinal2 = evaluate_cardinal_Bspline_basis(degree, tt)
monomial2 = cardinal2 @ Cinv.T

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
fig.suptitle("From cardinal to monomial")

for k in range(degree + 1):
    ax1.plot(tt, monomial2[:, k], label=f"t^{k}")
ax1.set_ylabel("Monomial basis")
ax1.set_title(f"Monomial basis polynomials (degree={degree})")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, cardinal2[:, k], label=f"C_{k}")
ax2.set_xlabel("t")
ax2.set_ylabel("Cardinal basis")
ax2.set_title(f"cardinal basis polynomials (degree={degree})")
ax2.legend()

plt.tight_layout()


plt.show()
