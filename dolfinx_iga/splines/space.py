from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import cpp as dlf_cpp
from dolfinx.common import IndexMap
from dolfinx.fem.function import FunctionSpace, _create_dolfinx_element

from dolfinx_iga.splines.element import create_cardinal_Bspline_element
from dolfinx_iga.splines.mesh import check_matching_mesh

if typing.TYPE_CHECKING:
    from basix.ufl import _ElementBase as ElementBase
    from dolfinx.fem import FunctionSpace
    from dolfinx.mesh import Mesh

    from dolfinx_iga.splines.bspline import Bspline
    from dolfinx_iga.splines.mesh import SplineMesh


def _get_owned_and_ghost_tp_dofs(
    mesh: SplineMesh,
    spline: Bspline,
) -> tuple[npt.NDArray[np.int64, np.int64], npt.NDArray[np.int64, np.int64]]:
    tp_cells = _get_tp_cells(mesh)
    assert False, "Not implemented"


def _get_local_to_global_dofs(
    mesh: SplineMesh,
    spline: Bspline,
    owned_dofs: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], np.int64]:
    assert False, "Not implemented"


def _get_ghost_global_dofs(
    mesh: SplineMesh,
    spline: Bspline,
    local_to_global_owned_dofs: npt.NDArray[np.int64],
    ghost_tp_dofs: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    assert False, "Not implemented"


def _create_cells_dofmap(
    mesh: SplineMesh, spline: Bspline, index_map: IndexMap
) -> dlf_cpp.graph.AdjacencyList_int64:
    assert False, "Not implemented"


def _create_dofmap(
    mesh: SplineMesh, spline: Bspline, cpp_element: dlf_cpp.fem.Element
) -> dlf_cpp.fem.DofMap:
    owned_tp_dofs, ghost_tp_dofs = _get_owned_and_ghost_tp_dofs(mesh, spline)
    local_to_global_owned_dofs, global_dof_offset = _get_local_to_global_dofs(
        mesh, spline, owned_tp_dofs
    )
    ghost_global_dofs, ghost_owners = _get_ghost_global_dofs(
        mesh, spline, local_to_global_owned_dofs, ghost_tp_dofs
    )

    num_local_dofs = local_to_global_owned_dofs.size
    index_map = IndexMap(
        mesh.comm,
        num_local_dofs,
        ghost_global_dofs,
        ghost_owners,
    )

    # Do the ghost get renumbered inside index_map?

    cells_dofmap = _create_cells_dofmap(mesh, spline, index_map)

    index_map_bs = 1  # So far
    bs = 1  # So far

    dof_layout = dlf_cpp.fem.create_element_dof_layout(cpp_element, [])

    return dlf_cpp.fem.DofMap(dof_layout, index_map, index_map_bs, cells_dofmap, bs)


class SplineFunctionSpace(FunctionSpace):
    def __init__(
        self,
        mesh: SplineMesh,
        spline: Bspline,
    ):
        self._check_input_validity(mesh, spline)

        self._spline = spline

        element = create_cardinal_Bspline_element(spline.degrees)
        cpp_space = self._create_cpp_space(mesh, element)

        super().__init__(mesh, element, cpp_space)

    def _check_input_validity(self, mesh: SplineMesh, spline: Bspline) -> None:
        if mesh.dim != spline.dim:
            raise ValueError("Mesh and spline dimensions do not match.")
        if mesh.dim != 1:
            raise ValueError("So far, only 1D splines are supported.")
        if mesh.geometry.x.dtype != spline.dtype:
            raise ValueError("Mesh and element dtype are not compatible.")

        if not check_matching_mesh(mesh, spline):
            raise ValueError("Mesh and spline do not match.")

    def _create_cpp_space(
        self, mesh: SplineMesh, element: ElementBase
    ) -> dlf_cpp.fem.FunctionSpace:
        cpp_element = _create_dolfinx_element(
            mesh.comm, mesh.topology.cell_type, element, self.dtype
        )

        cpp_dofmap = _create_dofmap(mesh, self.spline, cpp_element)

        # Initialize the cpp.FunctionSpace
        ufl_space = ufl.FunctionSpace(mesh.ufl_domain(), element)
        value_shape = ufl_space.value_shape

        if self.dtype == np.float64:
            space_creator = dlf_cpp.fem.FunctionSpace_float64
        else:
            space_creator = dlf_cpp.fem.FunctionSpace_float32

        return space_creator(mesh._cpp_object, cpp_element, cpp_dofmap, value_shape)

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def spline(self) -> Bspline:
        return self._spline

    @property
    def dtype(self) -> np.dtype:
        return self._spline.dtype


def create_spline_space(mesh: Mesh, spline: Bspline) -> SplineFunctionSpace:
    """Create a spline finite element function space.

    Args:
        mesh: Mesh that space is defined on.
        splines: List of splines that define the space.

    Returns:
        A function space.
    """

    return SplineFunctionSpace(mesh, spline)
