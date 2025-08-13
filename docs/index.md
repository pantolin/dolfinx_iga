# dolfinx_iga Documentation

## Installation

### For Development

1. Clone the repository:
```bash
git clone https://github.com/your-username/dolfinx_iga.git
cd dolfinx_iga
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .
```

### FEniCS Dependencies

Note: This package requires DOLFINx and related FEniCS components. These are typically installed via conda:

```bash
conda install -c conda-forge dolfinx basix ffcx ufl
```

## Running Tests

```bash
pytest tests/
```

Or with coverage:
```bash
pytest tests/ --cov=dolfinx_iga
```

## Examples

See the `examples/` directory for usage examples:

```bash
python examples/basic_bspline_curve.py
```

## API Reference

### B-spline Curves

```python
from dolfinx_iga import BSplineCurve
import numpy as np

# Define control points
control_points = np.array([[0, 0], [1, 2], [3, 1], [4, 0]])

# Create curve
curve = BSplineCurve(control_points, degree=3)

# Evaluate at parameter values
points = curve.evaluate(np.linspace(0, 1, 100))
```

### NURBS Curves

```python
from dolfinx_iga import NURBSCurve
import numpy as np

# Define control points and weights
control_points = np.array([[0, 0], [1, 2], [3, 1], [4, 0]])
weights = np.array([1.0, 2.0, 1.0, 1.0])

# Create NURBS curve
curve = NURBSCurve(control_points, weights, degree=3)

# Evaluate
points = curve.evaluate(np.linspace(0, 1, 100))
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request
