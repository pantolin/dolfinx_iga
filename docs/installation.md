# Installation

## Recommended: Conda Environment

The best way to install dolfinx_iga is using conda, which handles all FEniCS dependencies automatically.

### Quick Start
```bash
# Clone repository
git clone https://github.com/pantolin/dolfinx_iga.git
cd dolfinx_iga

# Create environment
conda env create -f environment.yml
conda activate dolfinx_iga

# Install package
pip install -e .
```

### Manual Conda Setup
```bash
# Create environment with FEniCS
conda create -n dolfinx_iga -c conda-forge python=3.11
conda activate dolfinx_iga
conda install -c conda-forge fenics-dolfinx fenics-basix fenics-ffcx fenics-ufl numpy

# Install dolfinx_iga
pip install -e .
```

## Requirements

### Core Dependencies (via conda-forge)
- Python >= 3.9
- fenics-dolfinx
- fenics-basix
- fenics-ffcx
- fenics-ufl
- numpy

### Optional Dependencies
- numba (for performance acceleration)
- matplotlib (for visualization examples)
- jupyter (for interactive examples)

## Alternative: PyPI + Conda

If you prefer pip for the main package:

```bash
# Install FEniCS via conda
conda install -c conda-forge fenics-dolfinx

# Install dolfinx_iga via pip (when available)
pip install dolfinx-iga
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
