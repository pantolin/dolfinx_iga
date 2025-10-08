import basix
import basix.ufl
import numpy as np
from basix import CellType

from dolfinx_iga.splines.basis_1D import evaluate_cardinal_Bspline_basis

BasixUFLElementBase = basix.ufl._ElementBase


class SplineElement:
    def __init__(self, degree, supdegree=None):
        if supdegree is None:
            supdegree = degree

        if degree < 0:
            raise ValueError("Degree must be non-negative")
        if supdegree < degree:
            raise ValueError("Supdegree must be no smaller than degree")

        self._degree = degree
        self._supdegree = supdegree
        self._element = self._create_cardinal_bspline_element_1D(degree, supdegree)

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def supdegree(self) -> int:
        return self._supdegree

    @property
    def basix_ufl_element(self) -> BasixUFLElementBase:
        return self._element

    def tabulate(self, *args, **kwargs):
        """Delegate to the underlying basix element."""
        return self._element.tabulate(*args, **kwargs)

    def __getattr__(self, name):
        """Automatically delegate any other methods/attributes to the basix element."""
        return getattr(self._element, name)

    @staticmethod
    def _create_cardinal_bspline_element_1D(
        degree: int, supdegree: int
    ) -> BasixUFLElementBase:
        n_basis = degree + 1
        n_supbasis = supdegree + 1

        wcoeffs = np.zeros((n_basis, n_supbasis), dtype=np.float64)
        wcoeffs[:, :n_basis] = np.eye(n_basis, dtype=np.float64)

        int_pts = [[], [], [], []]
        int_pts_interior, _ = basix.make_quadrature(CellType.interval, 2 * n_basis - 1)
        for _ in range(2):
            int_pts[0].append(np.zeros((0, 1), dtype=np.float64))
        int_pts[1].append(int_pts_interior)

        legendre_int = basix.tabulate_polynomials(
            basix.PolynomialType.legendre,
            CellType.interval,
            supdegree,
            int_pts_interior,
        )

        total_degree = degree + supdegree
        quad_pts, quad_wts = basix.make_quadrature(CellType.interval, total_degree)

        legendre = basix.tabulate_polynomials(
            basix.PolynomialType.legendre, CellType.interval, supdegree, quad_pts
        )

        splines = evaluate_cardinal_Bspline_basis(degree, quad_pts[:, 0])

        M1 = np.linalg.inv(legendre_int.T @ (legendre * quad_wts) @ splines)

        M = [[], [], [], []]
        M[0].append(np.zeros((0, 1, 0, 1), dtype=np.float64))
        M[0].append(np.zeros((0, 1, 0, 1), dtype=np.float64))
        M[1].append(M1.reshape(n_basis, 1, n_basis, 1))

        return basix.ufl.custom_element(
            CellType.interval,
            [],
            wcoeffs,
            int_pts,
            M,
            0,
            basix.MapType.identity,
            basix.SobolevSpace.H1,
            True,  # discontinuous
            degree,
            supdegree,
            basix.PolysetType.standard,
        )


# class SplineElement(FiniteElement):
#     def __new__(cls, degree, supdegree=None):
#         if supdegree is None:
#             supdegree = degree
#         # Create the basix element
#         element = create_cardinal_bspline_element_1D(degree, supdegree)
#         # Replace the element's class with our subclass
#         element.__class__ = cls
#         # Store extra attributes
#         element._degree = degree
#         element._supdegree = supdegree
#         return element

#     @property
#     def degree(self) -> int:
#         return self._degree

#     @property
#     def supdegree(self) -> int:
#         return self._supdegree
