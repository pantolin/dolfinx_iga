import basix
import numpy as np
from basix import CellType, ElementFamily, LagrangeVariant


def compare_lagrange_variants():
    """Compare different Lagrange variants for various elements."""

    variants = [
        LagrangeVariant.gll_centroid,
        LagrangeVariant.gll_warped,
        LagrangeVariant.gll_isaac,
    ]

    variant_names = ["gll_centroid", "gll_warped", "gll_isaac"]

    # Test cases: (cell_type, degree)
    test_cases = [
        (CellType.interval, 3),
        (CellType.triangle, 2),
        (CellType.quadrilateral, 2),
    ]

    for cell_type, degree in test_cases:
        print(f"\n{cell_type.name.upper()} element (degree={degree}):")
        print("=" * 80)

        elements = []
        for variant, name in zip(variants, variant_names):
            try:
                lagrange = basix.create_element(
                    ElementFamily.P, cell_type, degree, variant
                )
                elements.append((name, lagrange))
                print(f"\n{name}:")
                print(f"  Points shape: {lagrange.points.shape}")
                print(f"  DOF count: {lagrange.dim}")
                print(f"  First few points: {lagrange.points[:3]}")
            except Exception as e:
                print(f"\n{name}: Error - {e}")

        # Compare DOF ordering if we have multiple elements
        if len(elements) > 1:
            print("\nDOF Ordering Comparison:")
            for i, (name, element) in enumerate(elements):
                print(f"  {name}: {element.dof_ordering}")

        # Test tabulation at a few points
        if len(elements) > 0:
            test_points = (
                np.array([[0.5, 0.5]])
                if cell_type != CellType.interval
                else np.array([[0.5]])
            )
            print(f"\nTabulation at {test_points}:")
            for name, element in elements:
                try:
                    tab = element.tabulate(0, test_points)
                    print(
                        f"  {name}: shape {tab.shape}, first value {float(tab[0, 0, 0]):.6f}"
                    )
                except Exception as e:
                    print(f"  {name}: tabulation error - {e}")

    print("\n" + "=" * 80)
    print("Note: All variants use Gauss-Lobatto-Legendre (GLL) points")
    print("but may differ in internal implementation details.")

    # Check available variants
    print("\nAvailable LagrangeVariant options:")
    for variant in dir(LagrangeVariant):
        if not variant.startswith("_"):
            print(f"  {variant}")


if __name__ == "__main__":
    compare_lagrange_variants()
