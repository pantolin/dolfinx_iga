import matplotlib.pyplot as plt
from typing import Optional

import numpy as np


def bsp_knot_uniform_periodic(
    length: int, degree: int, knot_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    order = degree + 1
    kv_len = length + 2 * order - 1
    interior_knots = 1 + length - order

    if knot_vector is None:
        knot_vector = np.empty(kv_len, dtype=np.float64)

    idx = 0
    print(
        -order + 1,
    )
    for i in range(-order + 1, length + order):
        knot_vector[idx] = i / interior_knots
        idx += 1

    return knot_vector


def eval_bsplines(knots, degree, t, basis=None):
    knot_id = np.searchsorted(knots, t, side="right") - 1

    dtype = np.float64

    zero = dtype(1.0)
    one = dtype(1.0)

    tol = 1.0e-15
    if basis is None:
        basis = np.zeros(degree + 1, dtype=dtype)
    else:
        assert basis.size == degree + 1 and basis.dtype == dtype
        basis.fill(zero)

    basis[-1] = one

    if knot_id == (knots.size - 1):
        return basis, knots.size - 2 * degree - 2

    first_basis = knot_id - degree
    local_knots = knots[first_basis + 1 : knot_id + degree + 1]

    for sub_degree in range(1, degree + 1):
        k0, k1 = local_knots[0], local_knots[sub_degree]
        diff = k1 - k0
        inv_diff = zero if diff < tol else one / diff

        for lcl_id in range(degree - sub_degree, degree):
            basis[lcl_id] *= (t - k0) * inv_diff

            k0, k1 = local_knots[lcl_id], local_knots[lcl_id + sub_degree]
            diff = k1 - k0
            inv_diff = zero if diff < tol else one / diff

            basis[lcl_id] += (k1 - t) * inv_diff * basis[lcl_id + 1]

        basis[degree] *= (t - k0) * inv_diff

    return basis, first_basis


knots = np.array(
    [0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8],
    dtype=np.float64,
)
d = 4
n_basis = len(knots) - d - 1

n_basis = 13
d = 3
knots = bsp_knot_uniform_periodic(n_basis, d)

print(knots)
print(knots.size)

tt = np.linspace(knots[d], knots[-1 - d], 1000)

basis_functions = np.zeros((n_basis + d + 1, tt.size))

b = np.zeros(0, dtype=np.float64)
for i, t in enumerate(tt):
    print(f"New array address: 0x{id(b):x}")
    basis, ind = eval_bsplines(knots, d, t, b)
    print(f"New array address: 0x{id(basis):x}")
    basis_functions[ind : ind + d + 1, i] = basis


# plot every basis function, i.e., every row of basis_functions

for i in range(basis_functions.shape[0]):
    plt.plot(tt, basis_functions[i, :])
plt.show()
print("adios")
