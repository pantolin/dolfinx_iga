"""Example usage of tolerance utilities in dolfinx_iga."""

import numpy as np

from dolfinx_iga.knots import KnotsVector
from dolfinx_iga.utils.tolerance import (
    get_default_tolerance,
    get_tolerance_info,
    unique_with_tolerance,
)


def main():
    """Demonstrate tolerance utilities usage."""
    print("=== DOLFINx IGA Tolerance Utilities Example ===\n")

    # 1. Getting tolerance values for different dtypes
    print("1. Default Tolerance Values:")
    for dtype in [np.float16, np.float32, np.float64]:
        tol = get_default_tolerance(dtype)
        print(f"   {dtype.__name__}: {tol}")
    print()

    # 2. Finding unique values with tolerance
    print("2. Unique Values with Tolerance:")
    knot_values = np.array(
        [
            0.0,
            0.0,
            0.0,  # Triple knot at 0
            0.25,
            0.25000001,  # Nearly identical knots
            0.5,
            0.5,  # Double knot at 0.5
            0.75,
            0.75000001,  # Nearly identical knots
            1.0,
            1.0,
            1.0,  # Triple knot at 1
        ],
        dtype=np.float32,
    )

    print(f"   Original knots: {len(knot_values)} values")

    # Default tolerance
    unique_default, counts_default = unique_with_tolerance(knot_values, "default")
    print(f"   Unique (default): {len(unique_default)} values")
    print(f"   Values: {unique_default}")
    print(f"   Counts: {counts_default}")

    # Strict tolerance
    unique_strict, counts_strict = unique_with_tolerance(knot_values, "strict")
    print(f"   Unique (strict): {len(unique_strict)} values")
    print()

    # 3. Using with KnotsVector class
    print("3. KnotsVector with Tolerance:")
    kv = KnotsVector(knot_values)

    print("   Multiplicities:")
    multiplicities = kv.multiplicities
    unique_knots = kv.unique_knots
    for knot, mult in zip(unique_knots, multiplicities):
        print(f"     Knot {knot:.6f}: multiplicity {mult}")
    print()

    # 4. Tolerance information
    print("4. Comprehensive Tolerance Information:")
    info = get_tolerance_info(np.float32)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2e}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
