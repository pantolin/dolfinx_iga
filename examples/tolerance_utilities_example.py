"""Example usage of tolerance utilities in dolfinx_iga."""

import numpy as np

from dolfinx_iga.knots import KnotsVector
from dolfinx_iga.utils.tolerance import (
    are_arrays_close,
    are_close,
    get_default_tolerance,
    get_strict_tolerance,
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

    # 2. Comparing floating-point values
    print("2. Floating-Point Comparisons:")
    val1, val2 = 1.0, 1.0000001

    for dtype in [np.float32, np.float64]:
        close_default = are_close(val1, val2, dtype, "default")
        close_strict = are_close(val1, val2, dtype, "strict")
        close_conservative = are_close(val1, val2, dtype, "conservative")

        print(f"   {dtype.__name__}: 1.0 vs 1.0000001")
        print(f"     Default: {close_default}")
        print(f"     Strict: {close_strict}")
        print(f"     Conservative: {close_conservative}")
    print()

    # 3. Array comparisons
    print("3. Array Comparisons:")
    arr1 = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    arr2 = np.array([0.0000001, 0.5000001, 1.0000001], dtype=np.float32)

    arrays_close = are_arrays_close(arr1, arr2, "default")
    print(f"   Arrays close (default): {arrays_close}")

    arrays_close_strict = are_arrays_close(arr1, arr2, "strict")
    print(f"   Arrays close (strict): {arrays_close_strict}")
    print()

    # 4. Finding unique values with tolerance
    print("4. Unique Values with Tolerance:")
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

    # 5. Using with KnotsVector class
    print("5. KnotsVector with Tolerance:")
    kv = KnotsVector(knot_values)

    print("   Default multiplicities:")
    multiplicities = kv.multiplicities()
    for knot, mult in multiplicities.items():
        print(f"     Knot {knot:.6f}: multiplicity {mult}")

    print("   Strict multiplicities:")
    multiplicities_strict = kv.multiplicities(get_strict_tolerance(np.float32))
    for knot, mult in multiplicities_strict.items():
        print(f"     Knot {knot:.7f}: multiplicity {mult}")
    print()

    # 6. Tolerance information
    print("6. Comprehensive Tolerance Information:")
    info = get_tolerance_info(np.float32)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2e}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
