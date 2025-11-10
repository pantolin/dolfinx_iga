from __future__ import annotations

import typing

import basix.ufl
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import cpp as dlf_cpp
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import coordinate_element
from dolfinx.mesh import Mesh

if typing.TYPE_CHECKING:
    from mpi4py import MPI

    from dolfinx_iga.splines import Bspline1D
    from dolfinx_iga.splines.bspline import Bspline


class SplineMesh(Mesh):
    """A tensor-product spline mesh."""

    def __init__(self, comm: MPI.Comm, spline: Bspline):
        self._spline = spline
        self._dim = spline.dim

        if self._dim not in [1, 2, 3]:
            raise ValueError("Spline mesh must be 1D, 2D, or 3D.")

        self._unique_knots = [
            spline.get_unique_knots_and_multiplicity(in_domain=True)[0]
            for spline in spline.splines_1D
        ]

        self._create_dolfinx_mesh(comm)

    def _create_mesh_nodes(self, comm: MPI.Comm) -> None:
        if comm.rank == 0:
            n_pts = np.array([len(self._unique_knots[i]) for i in range(self._dim)])

            dim = self.dim

            if dim == 1:
                x = [self._unique_knots[0].copy()]
            elif dim == 2:
                x = np.meshgrid(
                    self._unique_knots[0], self._unique_knots[1], indexing="xy"
                )
            else:  # dim == 3
                x = np.meshgrid(
                    self._unique_knots[2],
                    self._unique_knots[1],
                    self._unique_knots[0],
                    indexing="ij",
                )[::-1]

            nodes = np.zeros((x[0].size, dim), dtype=x[0].dtype)
            for dir in range(dim):
                nodes[:, dir] = x[dir].ravel()

            nodes = nodes.reshape([*n_pts, dim])
        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could be achieved by distributing nodes and
            # cells since their creation.
            # See https://jsdokken.com/dolfinx_docs/meshes.html#mpi-communication
            # for further details.
            dtype = self._unique_knots[0].dtype
            nodes = np.empty((0, dim), dtype=dtype)

        return nodes

    def _create_mesh_connectivity(self, comm: MPI.Comm, n_pts: np.ndarray) -> None:
        n_nodes_per_cell = 2**self._dim

        if comm.rank == 0:
            assert len(n_pts) == self._dim, "Invalid number of points."

            n_cells = np.array(n_pts) - 1
            conn_first_cell = np.arange(n_nodes_per_cell, dtype=np.int64)

            conn = conn_first_cell + np.arange(n_cells[0], dtype=np.int64).reshape(
                n_cells[0], 1
            )
            if self._dim > 1:
                conn = conn.ravel() + np.arange(
                    0, n_pts[0] * n_cells[1], n_pts[0], dtype=np.int64
                ).reshape(n_cells[1], 1)
                if self._dim > 2:
                    conn = conn.ravel() + np.arange(
                        0,
                        n_pts[0] * n_pts[1] * n_cells[2],
                        n_pts[0] * n_pts[1],
                        dtype=np.int64,
                    ).reshape(n_cells[2], 1)

            conn = np.asarray(conn, dtype=np.int64, order="C").reshape(
                -1, n_nodes_per_cell
            )

        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could be achieved by distributing nodes and
            # cells since their creation.
            # See https://jsdokken.com/dolfinx_docs/meshes.html#mpi-communication
            # for further details.
            conn = np.empty((0, n_nodes_per_cell), dtype=np.int64, order="C")

        return conn

    def _create_dolfinx_mesh(self, comm: MPI.Comm) -> None:
        nodes = self._create_mesh_nodes(comm)
        n_pts = nodes.shape[:-1]

        dim = self._dim
        nodes = nodes.reshape(-1, dim)

        conn = self._create_mesh_connectivity(comm, n_pts)

        cell_type = [
            basix.CellType.interval,
            basix.CellType.quadrilateral,
            basix.CellType.hexahedron,
        ][dim - 1]

        element = basix.ufl.element(
            family="Lagrange", cell=cell_type, degree=1, shape=(dim,), dtype=nodes.dtype
        )
        domain = ufl.Mesh(element)

        e_ufl = domain.ufl_coordinate_element()
        cmap = coordinate_element(e_ufl.basix_element)

        ghost_mode = GhostMode.shared_facet
        if comm.size > 1:
            partitioner = dlf_cpp.mesh.create_cell_partitioner(ghost_mode)
        else:
            partitioner = None

        msh_cpp = dlf_cpp.mesh.create_mesh(
            comm, conn, cmap._cpp_object, nodes, partitioner
        )
        super().__init__(msh_cpp, domain)

    @property
    def spline(self) -> Bspline:
        """Get the multi-dimensional B-spline that defines the mesh."""
        return self._spline

    @property
    def dim(self) -> int:
        """Get the dimension of the spline mesh."""
        return self._dim

    @property
    def unique_knots(self) -> list[np.ndarray]:
        """Get the unique knots."""
        return self._unique_knots

    @property
    def num_local_cells(self) -> int:
        """Get the number of local cells in the mesh."""
        return self.topology.size(self.topology.dim)

    @property
    def num_cells_dir(self) -> tuple[int, ...]:
        """Get the number of cells in each dimension."""
        return tuple(len(knots) - 1 for knots in self.unique_knots)


def create_spline_mesh(comm: MPI.Comm, spline: Bspline) -> SplineMesh:
    """Create a spline mesh from a multi-dimensional B-spline.

    Args:
        comm (MPI.Comm): The MPI communicator to use.
        spline (Bspline): The multi-dimensional B-spline to create the mesh from.

    Returns:
        SplineMesh: The spline mesh created from the splines.
    """
    return SplineMesh(comm, spline)


def check_matching_mesh(mesh: SplineMesh, spline: Bspline) -> bool:
    """Check if the mesh matches the splines.

    This method checks if the mesh matches the splines by comparing the
    unique knots of the splines with the unique knots of the mesh.
    The tolerance associated to the 1D splines is used in this check.

    Args:
        mesh (SplineMesh): The mesh to check.
        spline (Bspline): The multi-dimensional B-spline to check.

    Returns:
        bool: True if the mesh matches the splines, False otherwise.
    """

    def _check_matching_mesh_1D(spline_1D_0: Bspline1D, spline_1D_1: Bspline1D) -> bool:
        knots_0 = spline_1D_0.get_unique_knots_and_multiplicity(in_domain=True)[0]
        knots_1 = spline_1D_1.get_unique_knots_and_multiplicity(in_domain=True)[0]
        return np.allclose(knots_0, knots_1, atol=spline_1D_0.tolerance)

    return all(
        _check_matching_mesh_1D(spline_1D_0, spline_1D_1)
        for spline_1D_0, spline_1D_1 in zip(spline.splines_1D, mesh.spline.splines_1D)
    )


def _compute_cells_midpoints(mesh: SplineMesh) -> npt.NDArray[np.float64 | np.float32]:
    """Compute the midpoints of the cells in the mesh.

    Args:
        mesh (SplineMesh): The spline mesh to compute the midpoints of.

    Returns:
        npt.NDArray[np.float64 | np.float32]: The midpoints of the cells in the mesh.
    """

    cell_points = mesh.geometry.x
    return cell_points.reshape(-1, mesh.dim, order="C").mean(axis=1)


def get_tp_cells(mesh: SplineMesh) -> npt.NDArray[np.int64]:
    """Return the tensor-product cell indices for all mesh cells owned by the current rank.

    The indices follow the Fortran column-major ordering.
    I.e., the first index is the fastest varying.

    Args:
        mesh (SplineMesh): The spline mesh to extract cell TP indices from.

    Returns:
        npt.NDArray[np.int64]: Array of flattened tensor-product cell indices owned by the current rank, one entry per local mesh cell.
    """

    cell_midpoints = _compute_cells_midpoints(mesh)

    # For each dimension, find which knot span the midpoint is in
    num_cells = cell_midpoints.shape[0]
    tp_indices = np.empty((num_cells, mesh.dim), dtype=np.int64)

    for d in range(mesh.dim):
        knots = mesh.unique_knots[d]
        # For each midpoint, find which knot span [knots[i], knots[i+1]) it is in
        # np.searchsorted returns the insertion index, so subtract 1
        indices = np.searchsorted(knots, cell_midpoints[:, d], side="right") - 1
        tp_indices[:, d] = indices

    # Now map tensor indices to column-major (Fortran ordering) flat indices
    multipliers = np.cumprod((1,) + mesh.num_cells_dir[:-1])
    flat_indices = np.dot(tp_indices, multipliers)

    return flat_indices
