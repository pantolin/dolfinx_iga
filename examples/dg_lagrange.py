import dolfinx
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Circumradius,
    FacetNormal,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
    div,
    dx,
    ds,
    dS,
    grad,
    inner,
    grad,
    avg,
    jump,
)

import ufl

N = 10
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, cell_type=dolfinx.mesh.CellType.quadrilateral
)
# mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, N, [0.0, 1.0])


from dolfinx_iga.splines.element import create_cardinal_Bspline_element


degrees = [2, 5]

el = create_cardinal_Bspline_element(
    degrees=degrees,  # NOTE: 4,5 gives weird interpolation estimates
    shape=(),
)

import numpy as np

V = dolfinx.fem.functionspace(mesh, el)
# V = dolfinx.fem.functionspace(mesh, ("DG", 2))

c_x = dolfinx.fem.Constant(mesh, 2 * np.pi)
c_y = dolfinx.fem.Constant(mesh, np.pi)


def u_exact(mod, x):
    return float(c_x) * x[0] ** 2
    # return mod.sin(float(c_x) * x[0]) * mod.cos(float(c_y) * x[1])


# uD = dolfinx.fem.Function(V)
# uD.interpolate(lambda x: u_exact(np, x))
# uD.x.scatter_forward()

x = SpatialCoordinate(mesh)
# error = inner(u_exact(ufl, x) - uD, u_exact(ufl, x) - uD) * dx
# L2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(error))
# gL2 = mesh.comm.allreduce(L2_error, op=MPI.SUM)
# print(f"L2 interpolation error: {numpy.sqrt(gL2)}")

f = -div(grad(u_exact(ufl, x)))
u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(mesh)

H = dolfinx.fem.functionspace(mesh, ("DG", 0))
h = dolfinx.fem.Function(H)
h.x.array[:] = mesh.h(
    mesh.topology.dim,
    np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local
        + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    ),
)


alpha = 10
gamma = 10
h_avg = avg(h)

a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
# Add DG/IP terms
a += -inner(avg(grad(v)), jump(u, n)) * dS - inner(jump(v, n), avg(grad(u))) * dS
a += +gamma / h_avg * inner(jump(v, n), jump(u, n)) * dS


# Add Nitsche terms
a += -inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx
L += (
    -inner(n, grad(v)) * u_exact(ufl, x) * ds
    + alpha / h * inner(u_exact(ufl, x), v) * ds
)

import dolfinx.fem.petsc

uh = dolfinx.fem.Function(V, name="uh")
problem = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    u=uh,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
)
uh = problem.solve()


error = inner(u_exact(ufl, x) - uh, u_exact(ufl, x) - uh) * dx
L2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(error))
gL2 = mesh.comm.allreduce(L2_error, op=MPI.SUM)
print(f"L2 solve error: {numpy.sqrt(gL2)}")

max_deg = max(degrees)
space = dolfinx.fem.functionspace(mesh, ("DG", max_deg))
u_out = dolfinx.fem.Function(space, name="uh")
u_out.interpolate(uh)
u_out.x.scatter_forward()
with dolfinx.io.VTXWriter(mesh.comm, "dg_lagrange.bp", [u_out]) as writer:
    writer.write(0.0)
