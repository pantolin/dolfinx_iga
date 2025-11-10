from mpi4py import MPI

from dolfinx_iga.splines.bspline import Bspline
from dolfinx_iga.splines.bspline_1D import Bspline1D
from dolfinx_iga.splines.knots import create_cardinal_Bspline_knot_vector
from dolfinx_iga.splines.mesh import create_spline_mesh
from dolfinx_iga.splines.space import create_spline_space

comm = MPI.COMM_WORLD
degree = 3

knots = create_cardinal_Bspline_knot_vector(10, degree)
spline_1D = Bspline1D(knots, degree)

spline = Bspline([spline_1D] * 1)
mesh = create_spline_mesh(comm, spline)
space = create_spline_space(mesh, spline)


# n_elem = 4
# Bsplines = [Bspline1D(n_elem, degree) for n_elem in n_elems]
# mesh = create_spline_mesh(MPI.COMM_WORLD, [n_elem], [degree])

# V = create_spline_space(mesh, spline)
print("hola")
# print(isinstance(V, SplineFunctionSpace))
# print("done!")
