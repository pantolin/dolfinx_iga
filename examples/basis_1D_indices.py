from dolfinx_iga.splines.bspline_1D import Bspline1D, Bspline1DDofsManager
from dolfinx_iga.splines.knots import (
    create_uniform_open_knot_vector,
)

num_intervals = 10
degree = 2
knots = create_uniform_open_knot_vector(num_intervals, degree, start=0.0, end=1.0)
spline = Bspline1D(knots, degree, periodic=True)

dofs_manager = Bspline1DDofsManager(spline)

cell_id = 3
local_basis_id = 1
global_basis_id = dofs_manager.get_global_basis_ids(cell_id, local_basis_id)
owner_cell_id = dofs_manager.get_first_cell_of_global_basis_id(global_basis_id)
first_global_basis_id_of_owner_cell = dofs_manager.get_first_global_basis_id_of_cell(
    owner_cell_id
)
local_basis_id_in_owner_cell = global_basis_id - first_global_basis_id_of_owner_cell

print("Input values:")
print(f"  cell_id: {cell_id}")
print(f"  local_basis_id: {local_basis_id}")
print("\nOutput values:")
print(f"  global_basis_id: {global_basis_id}")
print(f"  owner_cell_id: {owner_cell_id}")
print(f"  local_basis_id_in_owner_cell: {local_basis_id_in_owner_cell}")
