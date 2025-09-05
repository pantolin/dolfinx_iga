# Cox-de Boor Basis Function Evaluation

This document explains the Cox-de Boor algorithm implementation for evaluating B-spline basis functions in the `Bspline1D` class.

## Overview

The Cox-de Boor algorithm is a recursive method for evaluating B-spline basis functions. It's based on the recursive definition:

```
B_{i,0}(t) = 1 if t_i ≤ t < t_{i+1}, 0 otherwise

B_{i,k}(t) = (t - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(t) + (t_{i+k+1} - t)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(t)
```

## Implementation

The `evaluate_basis_functions` method in the `Bspline1D` class implements this algorithm:

```python
def evaluate_basis_functions(self, t: np.floating) -> tuple[npt.NDArray[np.floating], int]:
    """
    Evaluate B-spline basis functions using the Cox-de Boor algorithm.
    
    Args:
        t: Parameter value at which to evaluate the basis functions
        
    Returns:
        tuple containing:
            - basis_values: Array of non-zero basis function values (length = degree + 1)
            - first_index: Index of the first non-zero basis function
    """
```

## Key Features

1. **Efficient computation**: Only computes the non-zero basis functions (at most degree + 1)
2. **Numerical stability**: Uses tolerance-based comparisons for knot equality
3. **Domain validation**: Checks that the parameter is within the knot vector domain
4. **Boundary handling**: Properly handles evaluation at domain boundaries

## Properties Verified

The implementation maintains these B-spline properties:

- **Partition of Unity**: Basis functions sum to 1.0 at any parameter value
- **Non-negativity**: All basis function values are ≥ 0
- **Local Support**: At most (degree + 1) basis functions are non-zero
- **Continuity**: Proper handling of knot multiplicities

## Usage Example

```python
import numpy as np
from dolfinx_iga.bspline_1D import Bspline1D

# Create a quadratic B-spline
degree = 2
knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64)
bspline = Bspline1D(knots, degree)

# Evaluate basis functions at t = 1.5
t = np.float64(1.5)
basis_values, first_index = bspline.evaluate_basis_functions(t)

print(f"At t={t}:")
print(f"  First non-zero basis index: {first_index}")
print(f"  Basis values: {basis_values}")
print(f"  Sum: {np.sum(basis_values)}")  # Should be 1.0
```

## Comparison with Reference Implementation

This implementation is based on the IRIT `BspCrvCoxDeBoorBasis` function, providing:

- Similar algorithmic approach using the recursive Cox-de Boor formula
- Proper handling of domain boundaries and special cases
- Efficient computation focusing only on non-zero basis functions
- Robust numerical handling with tolerance-based comparisons

## Performance Considerations

The algorithm has O(degree²) complexity for each evaluation, making it suitable for:

- Single-point evaluations
- Interactive applications
- Educational purposes

For repeated evaluations at many points, consider pre-computing basis functions or using matrix-based approaches.
