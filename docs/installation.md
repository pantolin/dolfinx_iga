# Installation

## Requirements

### Core Dependencies
- Python >= 3.8
- numpy

### FEniCS Dependencies (recommended)
- dolfinx
- basix
- ffcx
- ufl

### Optional Dependencies
- numba (for performance acceleration)
- matplotlib (for visualization examples)

## Installation Methods

### From PyPI (when available)
```bash
pip install dolfinx-iga
```

### From Source
```bash
git clone https://github.com/pantolin/dolfinx_iga.git
cd dolfinx_iga
pip install -e .
```

### For Development
```bash
git clone https://github.com/pantolin/dolfinx_iga.git
cd dolfinx_iga
pip install -e .[dev]
```

## FEniCS Installation

The FEniCS dependencies (dolfinx, basix, ffcx, ufl) are best installed via conda:

```bash
conda install -c conda-forge dolfinx basix ffcx ufl
```

## Verification

Test your installation:

```python
import dolfinx_iga
print(f"dolfinx_iga version: {dolfinx_iga.__version__}")

# Run basic functionality test
import numpy as np
control_points = np.array([[0, 0], [1, 1], [2, 0]])
curve = dolfinx_iga.BSplineCurve(control_points, degree=2)
point = curve.evaluate(0.5)
print(f"Test evaluation: {point}")
```
