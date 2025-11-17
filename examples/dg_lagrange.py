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

gd = 1

num_elements_per_direction = [11, 3, 2]
if gd == 3:
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        *num_elements_per_direction,
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )
elif gd == 1:
    mesh = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD,
        num_elements_per_direction[0],
        [0.0, 1.0],
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )
else:
    raise RuntimeError("Only 1D and 3D meshes supported")

from dolfinx_iga.splines.element import create_cardinal_Bspline_element
import numpy as np


if gd == 1:
    degrees = [3]
else:
    degrees = [2, 3, 5]

el = create_cardinal_Bspline_element(
    degrees=degrees,  # NOTE: 4,5 gives weird interpolation estimates
    shape=(),
)
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
num_ghost_cells = mesh.topology.index_map(mesh.topology.dim).num_ghosts
num_cells = num_cells_local + num_ghost_cells
min_locs = np.zeros(3)
max_locs = np.zeros(3)
num_cells_per_direction = np.zeros(3, dtype=np.int32)
for i in range(mesh.geometry.dim):
    min_locs[i] = mesh.comm.allreduce(np.min(mesh.geometry.x[:, i]), op=MPI.MIN)
    max_locs[i] = mesh.comm.allreduce(np.max(mesh.geometry.x[:, i]), op=MPI.MAX)
    num_cells_per_direction[i] = num_elements_per_direction[i]

# Compute IJK index for each cell on the process
deltax = (max_locs - min_locs) / num_cells_per_direction
np.nan_to_num(deltax, copy=False)
midpoint = dolfinx.mesh.compute_midpoints(
    mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32)
)
ijk = np.ceil((midpoint - min_locs) / deltax) - 1  # IJK index in tensor product grid
np.nan_to_num(ijk, copy=False)
ijk = ijk.astype(np.int64)

# Collapse ijk index over x-y-z axis to unique_integer
multiplier = np.zeros(3, dtype=np.int32)
multiplier[: mesh.geometry.dim] = np.array(
    [np.prod(num_cells_per_direction[:i]) for i in range(mesh.geometry.dim)],
    dtype=np.int32,
)
global_idx = ijk @ multiplier.reshape(-1, 1).flatten()


dofs_global = [degrees[i] + num_cells_per_direction for i in range(mesh.geometry.dim)]
num_dofs_global = np.prod(dofs_global)

# Pick some dof ordering
# The dofs are currently ordered as [[x0,y0,z0], ..., [x0,y0,zN],
#  [x0,y1,z0], ... [x0,y1,zN], ..., [x0, yN, zN], [x1,y0,z0], ... [xN,yN,zN]]
# per element. We want the global dof ordering to be (z,y,x)

# First figure out local numbering of the dofs (and if they are owned)
# Currently make the first k+1 dofs owned by whoever owns the first element in that direction
# Cell j owns dof k+1 + j
# Also create map from local function (dof) to element number in 1D.
# NOTE: Currently assuming fixed support width for all dofs in a given direction
# Pablo will supply a better function for this later
num_dofs_per_cell = el.dim * el.block_size
global_dofs = np.empty((ijk.shape[0], num_dofs_per_cell), dtype=np.int64)
ijk_owner = np.empty((ijk.shape[0], num_dofs_per_cell), dtype=np.int32)
for cell, (i, j, k) in enumerate(ijk):
    if mesh.geometry.dim == 3:
        raise RuntimeError("bla")
    elif mesh.geometry.dim == 2:
        dof = int((int(i) + degrees[0]) * dofs_global[1] + (int(j) + degrees[1]))
    elif mesh.geometry.dim == 1:
        global_dofs[cell] = [i + j for j in range(degrees[0] + 1)]
        # First check if we are in first or last cell in tensor product grid.
        # If first cell, own first k+1 dofs in each direction
        # If last cell own the k+1 dof in each direction
        relative_cell_ownership_index = np.full(
            num_dofs_per_cell, np.inf, dtype=np.int32
        )
        indices = np.arange(len(relative_cell_ownership_index))[::-1]
        indices[indices > i] = i
        relative_cell_ownership_index[:] = i - indices
        relative_cell_ownership_index[-1] *= i != 0
        ijk_owner[cell] = relative_cell_ownership_index
    else:
        raise RuntimeError("Only 1,2,3D meshes supported")

print(f"{MPI.COMM_WORLD.rank}, {global_dofs=} {ijk=} {ijk_owner=}", flush=True)
exit()

# Need to determine which process has ownership of each that is not owned
ghosted_dofs_dm = np.isin(global_dofs, process_owned_dofs, invert=True)
find_indices = ijk[np.any(ghosted_dofs_dm, axis=1)]
ghosted_dofs = np.unique(global_dofs[ghosted_dofs_dm])

print(
    MPI.COMM_WORLD.rank,
    "Finding owners for indices:",
    find_indices,
    f"{process_owned_dofs=}",
    f"{ghosted_dofs=}",
    f"{num_cells_local=}",
    f"{ijk=}",
    global_dofs,
    flush=True,
)
assert len(ghosted_dofs) == 0
exit()
breakpoint()
# V = dolfinx.fem.functionspace(mesh, el)
# V = dolfinx.fem.functionspace(mesh, ("DG", 2))

c_x = dolfinx.fem.Constant(mesh, 2 * np.pi)
c_y = dolfinx.fem.Constant(mesh, np.pi)


def u_exact(mod, x):
    return float(c_x) * x[0] ** 2
    # return mod.sin(float(c_x) * x[0]) * mod.cos(float(c_y) * x[1])


uD = dolfinx.fem.Function(V)
uD.interpolate(lambda x: u_exact(np, x))
uD.x.scatter_forward()

x = SpatialCoordinate(mesh)
error = inner(u_exact(ufl, x) - uD, u_exact(ufl, x) - uD) * dx
L2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(error))
gL2 = mesh.comm.allreduce(L2_error, op=MPI.SUM)
print(f"L2 interpolation error: {numpy.sqrt(gL2)}")

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
L += -inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

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
