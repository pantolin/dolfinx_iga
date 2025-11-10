import dolfinx
import numpy as np
from mpi4py import MPI

from dolfinx_iga.splines.element import create_cardinal_Bspline_element

t = np.linspace(0, 1, 11)

element = create_cardinal_Bspline_element(degrees=[2, 3])

domain = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [1.0, 1.0]],
    [4, 4],
    cell_type=dolfinx.cpp.mesh.CellType.quadrilateral,
)
V = dolfinx.fem.functionspace(domain, element)
dofmap = V.dofmap
print("adios")
