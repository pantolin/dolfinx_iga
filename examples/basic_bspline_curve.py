"""
Basic B-spline curve example.

This example demonstrates how to create and evaluate a simple B-spline curve
using dolfinx_iga.
"""

import numpy as np
from dolfinx_iga import BSplineCurve

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def main():
    """Run the basic B-spline curve example."""
    
    # Define control points for a simple curve
    control_points = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 1.0],
        [4.0, 0.0],
        [5.0, 1.0]
    ])
    
    # Create B-spline curve with degree 3
    degree = 3
    curve = BSplineCurve(control_points, degree)
    
    print("Created B-spline curve:")
    print(f"  - Degree: {curve.degree}")
    print(f"  - Number of control points: {curve.n_control_points}")
    print(f"  - Knot vector: {curve.knot_vector}")
    
    # Evaluate curve at multiple parameter values
    u_values = np.linspace(0, 1, 100)
    curve_points = curve.evaluate(u_values)
    
    # Evaluate derivatives at same points
    derivatives = curve.derivative(u_values)
    
    # Create plot if matplotlib is available
    if HAS_MATPLOTLIB and plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot curve and control points
        ax1.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2, label='B-spline curve')
        ax1.plot(control_points[:, 0], control_points[:, 1], 'ro-', markersize=6, alpha=0.7, label='Control points')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('B-spline Curve')
        ax1.legend()
        ax1.axis('equal')
        
        # Plot derivative vectors
        skip = 10  # Plot every 10th derivative for clarity
        ax2.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=1, alpha=0.5, label='Curve')
        ax2.quiver(
            curve_points[::skip, 0], curve_points[::skip, 1],
            derivatives[::skip, 0], derivatives[::skip, 1],
            scale=20, width=0.003, alpha=0.7, color='red', label='Derivatives'
        )
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('B-spline Curve with Derivatives')
        ax2.legend()
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Matplotlib not available, skipping plot generation.")
    
    # Print some specific evaluations
    print("\nCurve evaluation examples:")
    print(f"  - At u=0.0: {curve.evaluate(0.0)}")
    print(f"  - At u=0.5: {curve.evaluate(0.5)}")
    print(f"  - At u=1.0: {curve.evaluate(1.0)}")
    
    print("\nDerivative evaluation examples:")
    print(f"  - At u=0.0: {curve.derivative(0.0)}")
    print(f"  - At u=0.5: {curve.derivative(0.5)}")
    print(f"  - At u=1.0: {curve.derivative(1.0)}")


if __name__ == "__main__":
    main()
