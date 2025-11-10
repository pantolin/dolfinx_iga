import dolfinx
import dolfinx.fem.petsc
import dolfinx.io
from mpi4py import MPI

# ## 1. Create a quadrilateral mesh
# We use dolfinx.mesh.create_rectangle to create a 2D mesh.
# By setting cell_type=dolfinx.mesh.CellType.quadrilateral, we get a quad mesh.

L = 1.0  # Length of the beam
H = 0.2  # Height of the beam
nx = 20  # Number of elements in x-direction
ny = 5  # Number of elements in y-direction

mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (L, H)),
    n=(nx, ny),
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)

# ## 2. Define the Finite Element Function Space
# We define a vector function space (V) using Lagrange elements of degree 1.
# dolfinx automatically handles the element type based on the mesh's cell shape.
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))

dofmap = V.dofmap
kk = dofmap.cell_dofs(0)
print(dofmap.index_map.links(0))
total_dofs = V.dofmap.index_map.size_global
print(dofmap)

# # ## 3. Define Boundary Conditions
# # The beam is clamped on the left end (x=0).


# # Find facets (edges) on the left boundary
# def left_boundary(x):
#     return np.isclose(x[0], 0.0)


# left_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim=1, marker=left_boundary)

# # Find degrees of freedom (dofs) corresponding to these facets
# left_dofs = dolfinx.fem.locate_dofs_topological(V, 1, left_facets)

# # Define the Dirichlet boundary condition (u = (0, 0) at x=0)
# u_zero = np.array([0, 0], dtype=ScalarType)
# bc = dolfinx.fem.dirichletbc(u_zero, left_dofs, V)

# # ## 4. Define the Variational Problem (Weak Form)
# # We define the weak form for plane stress elasticity.

# # Elasticity parameters
# E = 1.0e9  # Young's modulus
# nu = 0.3  # Poisson's ratio
# mu = E / (2.0 * (1.0 + nu))
# lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Plane strain
# # For plane stress, lambda is re-calculated:
# lmbda = 2.0 * mu * lmbda / (lmbda + 2.0 * mu)


# # Strain tensor (epsilon)
# def epsilon(u):
#     return ufl.sym(ufl.grad(u))


# # Stress tensor (sigma)
# def sigma(u):
#     return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2.0 * mu * epsilon(u)


# # Define trial and test functions
# u = ufl.TrialFunction(V)
# v = ufl.TestFunction(V)

# # Bilinear form (a)
# a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx


# # Linear form (L) - applying a downward traction on the right end
# # We need to define the Neumann boundary (right side)
# def right_boundary(x):
#     return np.isclose(x[0], L)


# right_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim=1, marker=right_boundary)

# # Mark the right facets with a specific tag (e.g., 1)
# facet_indices = right_facets
# facet_markers = np.full_like(facet_indices, 1)
# facet_tag = dolfinx.mesh.meshtags(mesh, 1, facet_indices, facet_markers)

# # Define the integration measure 'ds' over the marked facets
# ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

# # Define the traction force (downward)
# T = dolfinx.fem.Constant(mesh, ScalarType((0.0, -1.0e5)))

# # Linear form L(v) = integral(T . v) ds
# L = ufl.dot(T, v) * ds(1)  # ds(1) integrates over facets tagged with 1

# # ## 5. Solve the Linear Problem
# # We assemble the system and solve for the displacement 'uh'.

# problem = dolfinx.fem.petsc.LinearProblem(
#     a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# )

# print("Solving linear elasticity problem...")
# uh = problem.solve()
# uh.name = "Displacement"
# print("Solve complete.")

# # ## 6. Save the Result
# # We save the solution to an XDMF file for visualization in ParaView.
# # We also save the mesh.

# try:
#     with dolfinx.io.XDMFFile(mesh.comm, "quad_elasticity.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(uh)
#     print("Solution saved to quad_elasticity.xdmf")
# except ImportError:
#     print("dolfinx.io.XDMFFile requires 'h5py' and 'meshio'.")
