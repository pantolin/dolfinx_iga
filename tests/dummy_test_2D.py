"""2D Poisson problem with DOLFINx: -Δu = f on [0,1]², u=0 on boundary"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


def solve_poisson_2d():
    """Solve -Δu = f with u=0 on boundary, f = 8π²sin(2πx)sin(2πy)

    Exact solution: u = sin(2πx)sin(2πy)
    """
    # Create 2D mesh on [0, 1]²
    n_elem = 4
    degree = 2
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, n_elem, n_elem, mesh.CellType.quadrilateral
    )

    # Define function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    dofmap = V.dofmap

    if domain.comm.rank >= 0:
        mesh_index_map = domain.topology.index_map(domain.topology.dim)
        print(f"MPI rank {domain.comm.rank}")
        print(f"Mesh index map: {mesh_index_map}")
        print(f"  Mesh index map size global: {mesh_index_map.size_global}")
        print(f"  Mesh index map size local: {mesh_index_map.size_local}")
        print(f"  Mesh index map ghosts: {mesh_index_map.ghosts}")
        print(f"  Mesh index map owners: {mesh_index_map.owners}")
        print(f"  Mesh index map local range: {mesh_index_map.local_range}")
        print()
        print(f"Dofmap list: {dofmap.list}")
        print(f"Dofmap index map: {dofmap.index_map}")
        print(f"  Dofmap index map size global: {dofmap.index_map.size_global}")
        print(f"  Dofmap index map ghost size local: {dofmap.index_map.size_local}")
        print(f"  Dofmap index map ghost: {dofmap.index_map.ghosts}")
        print(f"  Dofmap index map owners: {dofmap.index_map.owners}")
        print(f"  Dofmap index map local range: {dofmap.index_map.local_range}")
        print()
        print()
        print()

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Source term: f = 8π²sin(2πx)sin(2πy)
    x = ufl.SpatialCoordinate(domain)
    f = 8 * np.pi**2 * ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])

    # Weak form: a(u,v) = L(v)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # Dirichlet BCs: u = 0 on boundary
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0

    def boundary(x):
        # Boundary is where x=0, x=1, y=0, or y=1
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)),
        )

    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    problem = LinearProblem(a, L, bcs=[bc])
    uh = problem.solve()

    # Compute L2 error vs exact solution u_ex = sin(2πx)sin(2πy)
    x_coords = V.tabulate_dof_coordinates()
    u_exact = np.sin(2 * np.pi * x_coords[:, 0]) * np.sin(2 * np.pi * x_coords[:, 1])

    # Use MPI-aware error computation
    local_error_sq = np.sum((uh.x.array - u_exact) ** 2)
    local_n = len(uh.x.array)

    global_error_sq = domain.comm.allreduce(local_error_sq, op=MPI.SUM)
    global_n = domain.comm.allreduce(local_n, op=MPI.SUM)

    error_l2 = np.sqrt(global_error_sq / global_n)

    return uh, error_l2, domain


def plot_solution(uh, domain):
    try:
        import matplotlib.pyplot as plt

        V = uh.function_space
        x_coords = V.tabulate_dof_coordinates()

        # Create a grid for plotting
        n_pts = 100
        x_plot = np.linspace(0, 1, n_pts)
        y_plot = np.linspace(0, 1, n_pts)
        X, Y = np.meshgrid(x_plot, y_plot)
        u_exact_plot = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Plot numerical solution (scatter)
        sc1 = axes[0].scatter(
            x_coords[:, 0],
            x_coords[:, 1],
            c=uh.x.array,
            cmap="viridis",
            s=10,
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Numerical Solution")
        axes[0].set_aspect("equal")
        plt.colorbar(sc1, ax=axes[0])

        # Plot exact solution (contour)
        cont = axes[1].contourf(X, Y, u_exact_plot, levels=20, cmap="viridis")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Exact Solution")
        axes[1].set_aspect("equal")
        plt.colorbar(cont, ax=axes[1])

        plt.suptitle("2D Poisson: $-\\Delta u = 8\\pi^2\\sin(2\\pi x)\\sin(2\\pi y)$")
        plt.tight_layout()
        plt.show()

    except ImportError:
        pass


def plot_mesh(domain, V):
    try:
        import pyvista
        from dolfinx import plot

        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim, tdim)
        topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)

        # Ensure cell_types are set to VTK_QUAD for quadrilaterals
        import vtk

        VTK_QUAD = getattr(vtk, "VTK_QUAD", 9)  # fallback to 9 if not present

        # If all cells are quads, overwrite cell_types
        if (
            hasattr(domain, "topology")
            and domain.topology.cell_type.name == "quadrilateral"
        ):
            cell_types[:] = VTK_QUAD

        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.3)

        # Add global cell numbers as labels at cell centers
        # Get global cell indices (assuming contiguous global numbering)
        index_map = domain.topology.index_map(tdim)
        local_range = index_map.local_range
        global_cell_indices = np.arange(local_range[0], local_range[1])

        # Compute cell centers
        cell_centers = grid.cell_centers().points

        for idx, center in zip(global_cell_indices, cell_centers):
            plotter.add_point_labels(
                [center],
                [f"C{idx}"],
                font_size=12,
                point_color="red",
                text_color="red",
                shape_opacity=0.0,
                point_size=8,
            )

        # Build a mapping from global DOF to all (cell, local_index) pairs
        dofmap = V.dofmap
        num_cells_local = domain.topology.index_map(tdim).size_local

        # Map: global_dof -> list of (cell_idx, local_dof_idx)
        dof_to_cells = {}

        print(f"\nRank {domain.comm.rank} - Element-to-DOF connectivity:")

        for cell_idx in range(num_cells_local):
            cell_dofs = dofmap.cell_dofs(cell_idx)
            global_cell_idx = local_range[0] + cell_idx

            print(f"  Cell C{global_cell_idx}: DOFs = {cell_dofs.tolist()}")

            for local_dof_idx, global_dof in enumerate(cell_dofs):
                if global_dof not in dof_to_cells:
                    dof_to_cells[global_dof] = []
                dof_to_cells[global_dof].append((global_cell_idx, local_dof_idx))

        # Add global DOF numbers at DOF positions with local indices
        dof_coords_all = V.tabulate_dof_coordinates()
        dof_index_map = V.dofmap.index_map
        dof_local_range = dof_index_map.local_range
        num_owned = dof_index_map.size_local
        num_ghosts = dof_index_map.num_ghosts

        # Global DOF indices for owned DOFs
        global_dof_indices_owned = np.arange(dof_local_range[0], dof_local_range[1])

        # Global DOF indices for ghost DOFs
        ghost_global_indices = dof_index_map.ghosts

        print(f"\nRank {domain.comm.rank} - DOF summary:")
        print(f"  Owned DOFs: {num_owned} (global range: {dof_local_range})")
        print(f"  Ghost DOFs: {num_ghosts}")
        print(f"  Ghost global indices: {ghost_global_indices}")

        # Split coordinates into owned and ghost
        dof_coords_owned = dof_coords_all[:num_owned]
        dof_coords_ghost = dof_coords_all[num_owned : num_owned + num_ghosts]

        # Create point cloud for owned DOFs (blue)
        if len(dof_coords_owned) > 0:
            dof_points_owned = pyvista.PolyData(dof_coords_owned)
            plotter.add_mesh(
                dof_points_owned,
                color="blue",
                point_size=15,
                render_points_as_spheres=True,
            )

        # Create point cloud for ghost DOFs (orange)
        if len(dof_coords_ghost) > 0:
            dof_points_ghost = pyvista.PolyData(dof_coords_ghost)
            plotter.add_mesh(
                dof_points_ghost,
                color="orange",
                point_size=15,
                render_points_as_spheres=True,
            )

        # Add labels for owned DOFs
        for dof_idx, dof_pos in zip(global_dof_indices_owned, dof_coords_owned):
            # Build label: global index + local indices in each element
            label_parts = [f"{dof_idx}"]
            if dof_idx in dof_to_cells:
                for cell_idx, local_idx in dof_to_cells[dof_idx]:
                    label_parts.append(f"C{cell_idx}[{local_idx}]")

            label = "\n".join(label_parts)

            # Offset label position slightly for visibility
            label_pos = dof_pos.copy()
            label_pos[2] = 0.01  # Slightly above mesh

            plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=10,
                text_color="darkblue",
                shape_opacity=0.0,
                point_size=0,
                always_visible=True,
            )

        # Add labels for ghost DOFs
        for dof_idx, dof_pos in zip(ghost_global_indices, dof_coords_ghost):
            # Build label: global index + local indices in each element
            label_parts = [f"{dof_idx} (ghost)"]
            if dof_idx in dof_to_cells:
                for cell_idx, local_idx in dof_to_cells[dof_idx]:
                    label_parts.append(f"C{cell_idx}[{local_idx}]")

            label = "\n".join(label_parts)

            # Offset label position slightly for visibility
            label_pos = dof_pos.copy()
            label_pos[2] = 0.01  # Slightly above mesh

            plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=10,
                text_color="darkorange",
                shape_opacity=0.0,
                point_size=0,
                always_visible=True,
            )

        plotter.view_xy()
        if not pyvista.OFF_SCREEN and domain.comm.size == 1:
            print("Showing mesh")
            plotter.show()
        else:
            print("Saving mesh to file (off-screen)")
            plotter.show(
                auto_close=False,
                interactive=False,
                window_size=[800, 800],
                # off_screen=True,
            )
            plotter.screenshot(f"fundamentals_mesh_rank_{domain.comm.rank}.png")
            plotter.close()

    except ImportError:
        pass


if __name__ == "__main__":
    uh, error_l2, domain = solve_poisson_2d()

    if domain.comm.rank == 0:
        print("Solved 2D Poisson: -Δu = 8π²sin(2πx)sin(2πy)")
        print(f"L2 error vs exact: {error_l2:.6e}")
        print(f"Running on {domain.comm.size} MPI process(es)")

    # Optional: plot if matplotlib available
    # Note: Each process plots its own mesh elements in separate windows
    # Set PLOT_ALL_RANKS=True to see mesh decomposition across all processes
    # PLOT_ALL_RANKS = False
    # if PLOT_ALL_RANKS or domain.comm.rank == 0:
    # plot_solution(uh, domain)
    V = uh.function_space
    plot_mesh(domain, V)
