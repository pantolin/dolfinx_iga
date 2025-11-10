"""2D Stokes problem with DOLFINx: -∇·σ = f, ∇·u = 0 on [0,1]²"""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


def solve_stokes_2d():
    """Solve Stokes problem with manufactured solution

    -μΔu + ∇p = f
    ∇·u = 0

    Exact solution:
        u = [sin(πx)²sin(2πy), -sin(2πx)sin(πy)²]
        p = cos(πx)sin(πy)
    """
    # Create 2D mesh on [0, 1]²
    n_elem = 4
    degree_v = 2  # Velocity degree
    degree_p = 1  # Pressure degree
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, n_elem, n_elem, mesh.CellType.quadrilateral
    )

    # Define mixed function space (Taylor-Hood)
    # Velocity element (vector, degree 2)
    v_element = element("Lagrange", domain.topology.cell_name(), degree_v, shape=(2,))
    # Pressure element (scalar, degree 1)
    p_element = element("Lagrange", domain.topology.cell_name(), degree_p)
    # Mixed element
    mixed_el = mixed_element([v_element, p_element])

    W = fem.functionspace(domain, mixed_el)
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    if domain.comm.rank >= 0:
        mesh_index_map = domain.topology.index_map(domain.topology.dim)
        print(f"MPI rank {domain.comm.rank}")
        print(
            f"Mesh: {mesh_index_map.size_local} cells (local), {mesh_index_map.size_global} (global)"
        )
        print(f"Velocity space: {V.dofmap.index_map.size_global} DOFs")
        print(f"Pressure space: {Q.dofmap.index_map.size_global} DOFs")
        print(f"Mixed space: {W.dofmap.index_map.size_global} DOFs")
        print()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Viscosity
    mu = 1.0

    # Manufactured solution source term
    x = ufl.SpatialCoordinate(domain)

    # Exact solution (for source term computation)
    u_exact_expr = ufl.as_vector(
        [
            ufl.sin(np.pi * x[0]) ** 2 * ufl.sin(2 * np.pi * x[1]),
            -ufl.sin(2 * np.pi * x[0]) * ufl.sin(np.pi * x[1]) ** 2,
        ]
    )
    p_exact_expr = ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Source term f = -μΔu + ∇p
    f = -mu * ufl.div(ufl.grad(u_exact_expr)) + ufl.grad(p_exact_expr)

    # Weak form for Stokes
    a = (
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    # Dirichlet BCs: u = u_exact on boundary
    def boundary(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)),
        )

    # Create boundary condition using exact solution
    u_exact_bc = fem.Function(V)
    u_exact_bc.interpolate(
        lambda x: np.array(
            [
                np.sin(np.pi * x[0]) ** 2 * np.sin(2 * np.pi * x[1]),
                -np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]) ** 2,
            ]
        )
    )

    # Locate boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, boundary
    )

    boundary_dofs_V = fem.locate_dofs_topological(
        (W.sub(0), V),
        domain.topology.dim - 1,
        boundary_facets,
    )
    bc = fem.dirichletbc(u_exact_bc, boundary_dofs_V, W.sub(0))

    # Solve
    problem = LinearProblem(a, L, bcs=[bc])
    wh = problem.solve()

    # Extract velocity and pressure
    uh = fem.Function(V)
    uh.x.array[:] = wh.x.array[V_to_W]

    ph = fem.Function(Q)
    ph.x.array[:] = wh.x.array[Q_to_W]

    # Normalize pressure (set mean to zero)
    p_mean_local = np.sum(ph.x.array)
    p_count_local = len(ph.x.array)
    p_mean_global = domain.comm.allreduce(p_mean_local, op=MPI.SUM)
    p_count_global = domain.comm.allreduce(p_count_local, op=MPI.SUM)
    p_mean = p_mean_global / p_count_global
    ph.x.array[:] -= p_mean

    # Compute errors
    u_coords = V.tabulate_dof_coordinates()
    u_exact_vals = np.array(
        [
            np.sin(np.pi * u_coords[:, 0]) ** 2 * np.sin(2 * np.pi * u_coords[:, 1]),
            -np.sin(2 * np.pi * u_coords[:, 0]) * np.sin(np.pi * u_coords[:, 1]) ** 2,
        ]
    ).T

    # Reshape uh values for vector comparison
    uh_vals = uh.x.array.reshape(-1, 2)

    local_u_error_sq = np.sum((uh_vals - u_exact_vals) ** 2)
    local_u_n = len(uh_vals)

    global_u_error_sq = domain.comm.allreduce(local_u_error_sq, op=MPI.SUM)
    global_u_n = domain.comm.allreduce(local_u_n, op=MPI.SUM)

    error_u_l2 = np.sqrt(global_u_error_sq / global_u_n)

    # Pressure error
    p_coords = Q.tabulate_dof_coordinates()
    p_exact_vals = np.cos(np.pi * p_coords[:, 0]) * np.sin(np.pi * p_coords[:, 1])
    p_exact_vals -= np.mean(p_exact_vals)  # Normalize exact pressure too

    local_p_error_sq = np.sum((ph.x.array - p_exact_vals) ** 2)
    local_p_n = len(ph.x.array)

    global_p_error_sq = domain.comm.allreduce(local_p_error_sq, op=MPI.SUM)
    global_p_n = domain.comm.allreduce(local_p_n, op=MPI.SUM)

    error_p_l2 = np.sqrt(global_p_error_sq / global_p_n)

    return uh, ph, error_u_l2, error_p_l2, domain, V, Q


def plot_solution(uh, ph, domain):
    try:
        import matplotlib.pyplot as plt

        V = uh.function_space
        Q = ph.function_space
        u_coords = V.tabulate_dof_coordinates()
        p_coords = Q.tabulate_dof_coordinates()

        # Reshape velocity for plotting
        uh_vals = uh.x.array.reshape(-1, 2)

        # Create exact solutions for comparison
        n_pts = 50
        x_plot = np.linspace(0, 1, n_pts)
        y_plot = np.linspace(0, 1, n_pts)
        X, Y = np.meshgrid(x_plot, y_plot)

        U_exact = np.sin(np.pi * X) ** 2 * np.sin(2 * np.pi * Y)
        V_exact = -np.sin(2 * np.pi * X) * np.sin(np.pi * Y) ** 2
        P_exact = np.cos(np.pi * X) * np.sin(np.pi * Y)
        P_exact -= np.mean(P_exact)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Velocity magnitude - numerical
        u_mag = np.sqrt(uh_vals[:, 0] ** 2 + uh_vals[:, 1] ** 2)
        sc1 = axes[0, 0].scatter(
            u_coords[:, 0], u_coords[:, 1], c=u_mag, cmap="viridis", s=20
        )
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        axes[0, 0].set_title("Velocity Magnitude (Numerical)")
        axes[0, 0].set_aspect("equal")
        plt.colorbar(sc1, ax=axes[0, 0])

        # Velocity magnitude - exact
        u_mag_exact = np.sqrt(U_exact**2 + V_exact**2)
        cont1 = axes[0, 1].contourf(X, Y, u_mag_exact, levels=20, cmap="viridis")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        axes[0, 1].set_title("Velocity Magnitude (Exact)")
        axes[0, 1].set_aspect("equal")
        plt.colorbar(cont1, ax=axes[0, 1])

        # Velocity field with streamlines
        axes[0, 2].streamplot(
            X, Y, U_exact, V_exact, color="k", linewidth=0.5, density=1.5
        )
        axes[0, 2].quiver(
            u_coords[::2, 0],
            u_coords[::2, 1],
            uh_vals[::2, 0],
            uh_vals[::2, 1],
            color="r",
            alpha=0.6,
            scale=10,
        )
        axes[0, 2].set_xlabel("x")
        axes[0, 2].set_ylabel("y")
        axes[0, 2].set_title("Velocity Field (red=numerical, black=exact)")
        axes[0, 2].set_aspect("equal")

        # Pressure - numerical
        sc2 = axes[1, 0].scatter(
            p_coords[:, 0], p_coords[:, 1], c=ph.x.array, cmap="coolwarm", s=30
        )
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        axes[1, 0].set_title("Pressure (Numerical)")
        axes[1, 0].set_aspect("equal")
        plt.colorbar(sc2, ax=axes[1, 0])

        # Pressure - exact
        cont2 = axes[1, 1].contourf(X, Y, P_exact, levels=20, cmap="coolwarm")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("y")
        axes[1, 1].set_title("Pressure (Exact)")
        axes[1, 1].set_aspect("equal")
        plt.colorbar(cont2, ax=axes[1, 1])

        # Divergence check
        from dolfinx import fem as _fem

        div_form = ufl.div(uh) * ufl.dx
        divergence = _fem.assemble_scalar(_fem.form(div_form))
        divergence = domain.comm.allreduce(divergence, op=MPI.SUM)

        axes[1, 2].text(
            0.5,
            0.6,
            f"∫ ∇·u dx = {divergence:.2e}",
            ha="center",
            va="center",
            fontsize=14,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].text(
            0.5,
            0.4,
            "(should be ≈ 0)",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].set_title("Incompressibility Check")
        axes[1, 2].axis("off")

        plt.suptitle("2D Stokes Problem: Taylor-Hood Elements (P2-P1)")
        plt.tight_layout()
        plt.show()

    except ImportError:
        pass


def plot_mesh(domain, V, Q):
    """Plot mesh with DOF locations for both velocity and pressure"""
    try:
        import pyvista
        from dolfinx import plot

        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim, tdim)
        topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)

        # Ensure cell_types are set to VTK_QUAD for quadrilaterals
        import vtk

        VTK_QUAD = getattr(vtk, "VTK_QUAD", 9)
        if domain.topology.cell_type.name == "quadrilateral":
            cell_types[:] = VTK_QUAD

        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.3)

        # Add cell labels
        index_map = domain.topology.index_map(tdim)
        local_range = index_map.local_range
        global_cell_indices = np.arange(local_range[0], local_range[1])
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

        # Velocity DOFs (blue)
        v_coords_all = V.tabulate_dof_coordinates()
        v_dof_index_map = V.dofmap.index_map
        v_num_owned = v_dof_index_map.size_local
        v_num_ghosts = v_dof_index_map.num_ghosts

        # Extract only unique geometric positions (velocity is vector, so coords repeat)
        v_coords_unique = v_coords_all[::2]  # Every 2nd entry (x-component positions)
        v_num_owned_unique = v_num_owned // 2
        v_num_ghosts_unique = v_num_ghosts // 2

        v_coords_owned = v_coords_unique[:v_num_owned_unique]
        v_coords_ghost = v_coords_unique[
            v_num_owned_unique : v_num_owned_unique + v_num_ghosts_unique
        ]

        v_dof_local_range = v_dof_index_map.local_range
        v_global_dof_owned = (
            np.arange(v_dof_local_range[0], v_dof_local_range[1], 2) // 2
        )
        v_ghost_global = v_dof_index_map.ghosts[::2] // 2

        if len(v_coords_owned) > 0:
            v_points_owned = pyvista.PolyData(v_coords_owned)
            plotter.add_mesh(
                v_points_owned,
                color="blue",
                point_size=15,
                render_points_as_spheres=True,
            )

        if len(v_coords_ghost) > 0:
            v_points_ghost = pyvista.PolyData(v_coords_ghost)
            plotter.add_mesh(
                v_points_ghost,
                color="orange",
                point_size=15,
                render_points_as_spheres=True,
            )

        # Pressure DOFs (green)
        p_coords_all = Q.tabulate_dof_coordinates()
        p_dof_index_map = Q.dofmap.index_map
        p_num_owned = p_dof_index_map.size_local
        p_num_ghosts = p_dof_index_map.num_ghosts

        p_coords_owned = p_coords_all[:p_num_owned]
        p_coords_ghost = p_coords_all[p_num_owned : p_num_owned + p_num_ghosts]

        p_dof_local_range = p_dof_index_map.local_range
        p_global_dof_owned = np.arange(p_dof_local_range[0], p_dof_local_range[1])
        p_ghost_global = p_dof_index_map.ghosts

        if len(p_coords_owned) > 0:
            p_points_owned = pyvista.PolyData(p_coords_owned)
            plotter.add_mesh(
                p_points_owned,
                color="green",
                point_size=12,
                render_points_as_spheres=True,
            )

        if len(p_coords_ghost) > 0:
            p_points_ghost = pyvista.PolyData(p_coords_ghost)
            plotter.add_mesh(
                p_points_ghost,
                color="yellow",
                point_size=12,
                render_points_as_spheres=True,
            )

        # Add labels for velocity DOFs
        for i, (dof_idx, dof_pos) in enumerate(zip(v_global_dof_owned, v_coords_owned)):
            label = f"V{dof_idx}"
            label_pos = dof_pos.copy()
            label_pos[2] = 0.02
            plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=9,
                text_color="darkblue",
                shape_opacity=0.0,
                point_size=0,
                always_visible=True,
            )

        for dof_idx, dof_pos in zip(v_ghost_global, v_coords_ghost):
            label = f"V{dof_idx}(g)"
            label_pos = dof_pos.copy()
            label_pos[2] = 0.02
            plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=9,
                text_color="darkorange",
                shape_opacity=0.0,
                point_size=0,
                always_visible=True,
            )

        # Add labels for pressure DOFs
        for dof_idx, dof_pos in zip(p_global_dof_owned, p_coords_owned):
            label = f"P{dof_idx}"
            label_pos = dof_pos.copy()
            label_pos[2] = 0.01
            plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=9,
                text_color="darkgreen",
                shape_opacity=0.0,
                point_size=0,
                always_visible=True,
            )

        for dof_idx, dof_pos in zip(p_ghost_global, p_coords_ghost):
            label = f"P{dof_idx}(g)"
            label_pos = dof_pos.copy()
            label_pos[2] = 0.01
            plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=9,
                text_color="gold",
                shape_opacity=0.0,
                point_size=0,
                always_visible=True,
            )

        print(f"\nRank {domain.comm.rank} - Stokes DOF summary:")
        print(
            f"  Velocity DOFs (owned): {v_num_owned_unique}, ghosts: {v_num_ghosts_unique}"
        )
        print(f"  Pressure DOFs (owned): {p_num_owned}, ghosts: {p_num_ghosts}")

        plotter.view_xy()
        if not pyvista.OFF_SCREEN and domain.comm.size == 1:
            plotter.show()
        else:
            plotter.show(auto_close=False, interactive=False, window_size=[1000, 1000])
            plotter.screenshot(f"stokes_mesh_rank_{domain.comm.rank}.png")
            plotter.close()

    except ImportError as e:
        print(f"Could not plot: {e}")


if __name__ == "__main__":
    uh, ph, error_u_l2, error_p_l2, domain, V, Q = solve_stokes_2d()

    if domain.comm.rank == 0:
        print("Solved 2D Stokes problem with manufactured solution")
        print(f"Velocity L2 error: {error_u_l2:.6e}")
        print(f"Pressure L2 error: {error_p_l2:.6e}")
        print(f"Running on {domain.comm.size} MPI process(es)")

    # Plot solution
    # if domain.comm.rank == 0:
    #     plot_solution(uh, ph, domain)

    # Plot mesh with DOFs
    plot_mesh(domain, V, Q)
