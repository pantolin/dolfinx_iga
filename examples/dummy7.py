import matplotlib.pyplot as plt
import numpy as np

from dolfinx_iga.splines.basis_1D import (
    evaluate_Bernstein_basis,
    evaluate_cardinal_Bspline_basis,
)
from dolfinx_iga.splines.change_basis_1D import (
    create_Bernstein_to_cardinal_basis_operator,
    create_cardinal_to_Bernstein_basis_operator,
)

degree = 8
C = create_Bernstein_to_cardinal_basis_operator(degree)


n_samples = 1000
tt = np.linspace(0, 1, n_samples, dtype=np.float64)
bernstein = evaluate_Bernstein_basis(degree, tt)
cardinal = bernstein @ C.T


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
fig.suptitle("From Bernstein to cardinal")

for k in range(degree + 1):
    ax1.plot(tt, bernstein[:, k], label=f"B_{k}")
ax1.set_ylabel("Bernstein basis")
ax1.set_title(f"Bernstein basis polynomials (degree={degree})")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, cardinal[:, k], label=f"C_{k}")
ax2.set_xlabel("t")
ax2.set_ylabel("Cardinal basis")
ax2.set_title(f"Cardinal basis polynomials (degree={degree})")
ax2.legend()

plt.tight_layout()


Cinv = create_cardinal_to_Bernstein_basis_operator(degree)
cardinal2 = evaluate_cardinal_Bspline_basis(degree, tt)
bernstein2 = cardinal2 @ Cinv.T

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
fig.suptitle("From cardinal to Bernstein")

for k in range(degree + 1):
    ax1.plot(tt, bernstein2[:, k], label=f"B_{k}")
ax1.set_ylabel("Bernstein basis")
ax1.set_title(f"Bernstein basis polynomials (degree={degree})")
ax1.legend()

for k in range(degree + 1):
    ax2.plot(tt, cardinal2[:, k], label=f"C_{k}")
ax2.set_xlabel("t")
ax2.set_ylabel("Cardinal basis")
ax2.set_title(f"cardinal basis polynomials (degree={degree})")
ax2.legend()

plt.tight_layout()


plt.show()
