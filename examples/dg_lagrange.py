import dolfinx
import numpy
from dolfinx_iga.splines.element import create_cardinal_Bspline_element
import numpy as np
from dolfinx_iga.splines.bspline_1D import Bspline1D, Bspline1DDofsManager
from dolfinx_iga.splines.knots import (
    create_uniform_open_knot_vector,
)
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

gd = 2

num_elements_per_direction = [4, 2, 4]
if gd == 3:
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        *num_elements_per_direction,
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )
elif gd == 2:
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        *num_elements_per_direction[:2],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
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

if gd == 1:
    degrees = [2]
elif gd == 2:
    degrees = [2, 2]
else:
    degrees = [2, 3, 5]

el = create_cardinal_Bspline_element(
    degrees=degrees,  # NOTE: 4,5 gives weird interpolation estimates
    shape=(),
)

# As the mesh might have been created a-priori, we compute the extent in each direction.
# Of course this assumes that the mesh is structured in a tensor-product way.
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
num_ghost_cells = mesh.topology.index_map(mesh.topology.dim).num_ghosts
num_cells = num_cells_local + num_ghost_cells
min_locs = np.zeros(3)
max_locs = np.zeros(3)
num_cells_per_direction = np.zeros(3, dtype=np.int32)
# NOTE: Ensure that we handle 0 cells on process
for i in range(mesh.geometry.dim):
    min_locs[i] = mesh.comm.allreduce(np.min(mesh.geometry.x[:, i]), op=MPI.MIN)
    max_locs[i] = mesh.comm.allreduce(np.max(mesh.geometry.x[:, i]), op=MPI.MAX)
    num_cells_per_direction[i] = num_elements_per_direction[i]


# Create corresponding 1D splines on the same mesh.
spline_managers = []
for i in range(mesh.geometry.dim):
    knots = create_uniform_open_knot_vector(
        num_cells_per_direction[i], degrees[i], start=min_locs[i], end=max_locs[i]
    )
    spline = Bspline1D(knots, degrees[i], periodic=True)
    spline_managers.append(Bspline1DDofsManager(spline))

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
cell_multiplier = np.zeros(3, dtype=np.int32)
cell_multiplier[: mesh.geometry.dim] = np.array(
    [np.prod(num_cells_per_direction[:i]) for i in range(mesh.geometry.dim)],
    dtype=np.int64,
)
global_cell_idx = ijk @ cell_multiplier
num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global

# 1 if cell is owned and ghosted, 0 if ghost, -1 if not on process
# 2 if owned and not ghosted on other process
cell_on_process = np.full(num_cells_global, -1, dtype=np.int32)
cell_on_process[global_cell_idx[:num_cells_local]] = 1
cell_on_process[global_cell_idx[num_cells_local:]] = 0

# Check if cell is ghosted by any other process
cell_vec = dolfinx.la.vector(mesh.topology.index_map(mesh.topology.dim))
cell_vec.array[num_cells_local:] = 1
cell_vec.scatter_reverse(dolfinx.la.InsertMode.add)
cell_on_process[
    global_cell_idx[np.flatnonzero(cell_vec.array[:num_cells_local] == 0)]
] = 2


dofs_global = np.ones(3, dtype=np.int32)
dofs_global[: mesh.geometry.dim] = [
    degrees[i] + num_cells_per_direction[i] for i in range(mesh.geometry.dim)
]
num_dofs_global = np.prod(dofs_global)
dof_multiplier = np.zeros(3, dtype=np.int32)
dof_multiplier[: mesh.geometry.dim] = np.array(
    [np.prod(dofs_global[:i]) for i in range(mesh.geometry.dim)], dtype=np.int32
)


def dof_pos_to_global_index(dof_pos: tuple[int, ...]) -> int:
    assert (dof_pos < dofs_global).all()
    return dof_multiplier @ dof_pos


def invert_cell_idx(cell_idx: int) -> tuple[int, ...]:
    if mesh.geometry.dim == 1:
        return int(cell_idx)
    elif mesh.geometry.dim == 2:
        Nx = num_cells_per_direction[0]
        j = cell_idx // Nx
        i = cell_idx % Nx
        return int(i), int(j)
    elif mesh.geometry.dim == 3:
        Ny = num_cells_per_direction[1]
        Nz = num_cells_per_direction[2]
        k = cell_idx // (Ny * Nz)
        j = (cell_idx - k * Ny * Nz) // Nz
        i = cell_idx - k * Ny * Nz - j * Nz
        return int(i), int(j), int(k)
    else:
        raise RuntimeError("Only 1D, 2D and 3D supported")


# Function/Spline indicator
# -1 means not on process
# 1 means owned by process (and ghosted on other process)
# 0 means ghosted by process
# 2 means owned by process and not ghosted anywhere else
func_indicator = np.full(num_dofs_global, -1, dtype=np.int32)

tmp_ijk = np.zeros(3, dtype=np.int64)

print(MPI.COMM_WORLD.rank, f"{num_cells_local=} {num_ghost_cells=}", flush=True)

# For each process, determine which DOFs are owned.
dof_ownership = np.full(dofs_global[: mesh.geometry.dim], -1, dtype=np.int32)
for cell, l_ijk in enumerate(ijk):
    for dir_0 in range(mesh.geometry.dim):
        for dir_1 in range(dir_0 + 1, mesh.geometry.dim):
            # Note: at least vectorize for l_idx.
            for l0 in range(degrees[dir_0] + 1):
                global_func_idx_0 = spline_managers[dir_0].get_global_basis_ids(
                    l_ijk[dir_0], l0
                )
                cell0_idx = spline_managers[dir_0].get_first_cell_of_global_basis_id(
                    global_func_idx_0
                )

                for l1 in range(degrees[dir_1] + 1):
                    global_func_idx_1 = spline_managers[dir_1].get_global_basis_ids(
                        l_ijk[dir_1], l1
                    )
                    cell1_idx = spline_managers[
                        dir_1
                    ].get_first_cell_of_global_basis_id(global_func_idx_1)
                    tmp_ijk[:] = l_ijk
                    tmp_ijk[dir_0] = cell0_idx
                    tmp_ijk[dir_1] = cell1_idx
                    owns_dof = cell_on_process[tmp_ijk @ cell_multiplier]
                    dof_ownership[global_func_idx_0, global_func_idx_1] = owns_dof > 0

print(MPI.COMM_WORLD.rank, dof_ownership.T[::-1], flush=True)
# NOTE: This follows Pablos ordering of DOFs.
owned_dof_indices = np.vstack(np.nonzero(dof_ownership > 0)).T
owned_dof_indices_global = owned_dof_indices @ dof_multiplier[: mesh.geometry.dim]

num_owned_dofs_local = len(owned_dof_indices_global)
local_range_start = mesh.comm.exscan(num_owned_dofs_local, op=MPI.SUM)
local_range_start = 0 if local_range_start is None else local_range_start
local_range_end = local_range_start + num_owned_dofs_local
print(MPI.COMM_WORLD.rank, local_range_start, local_range_end, flush=True)

# print(MPI.COMM_WORLD.rank, owned_dof_indices_global)


exit()

# print(
#     f"Local cell {cell}, ijk={l_ijk}, axis={dir}, local_func_idx={l_idx}, global_func_idx={global_func_idx}"
# )
# for i in range(mesh.geometry.dim):
print(MPI.COMM_WORLD.rank, dof_ownership.T)
on_proc_cells = np.flatnonzero(cell_on_process > 0)
ijk_on_proc = [invert_cell_idx(on_proc_cell) for on_proc_cell in on_proc_cells]
print(MPI.COMM_WORLD.rank, ijk_on_proc)
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
