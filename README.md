# dolfinx_iga

A modern isogeometric analysis (IGA) extension for DOLFINx, built for the FEniCSx ecosystem.

## Overview

`dolfinx_iga` is a Python library that extends [DOLFINx](https://github.com/FEniCS/dolfinx) with isogeometric analysis capabilities. It provides B-spline and NURBS (rational B-spline) functionalities, inspired by [tIGAr](https://github.com/david-kamensky/tIGAr) but modernized for the current FEniCSx ecosystem.

## Features

- B-spline basis functions and geometry representation
- NURBS (rational B-spline) support
- Integration with DOLFINx finite element framework
- Efficient numerical computations with optional Numba acceleration
- Comprehensive test suite with pytest

## Installation

### Recommended: Using Conda

The easiest way to install dolfinx_iga with all dependencies:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dolfinx_iga

# Install dolfinx_iga in development mode
pip install -e .
```

### Alternative: Manual Conda Setup

```bash
# Create environment with FEniCS
conda create -n dolfinx_iga -c conda-forge python=3.11 fenics-dolfinx numpy
conda activate dolfinx_iga
pip install -e .
```

### From PyPI (when available)
```bash
pip install dolfinx-iga
# Note: Requires separate FEniCS installation via conda
```

### From source
```bash
git clone https://github.com/pantolin/dolfinx_iga.git
cd dolfinx_iga
pip install -e .
```

## Dependencies

### Required (via conda-forge)
- `fenics-dolfinx` - The core DOLFINx library
- `fenics-basix` - Finite element basis functions  
- `fenics-ffcx` - FEniCS Form Compiler
- `fenics-ufl` - Unified Form Language
- `numpy` - Numerical computing

### Optional
- `numba` - JIT compilation for performance optimization
- `matplotlib` - Plotting and visualization

## Quick Start

```python
import dolfinx_iga as iga
import numpy as np

# Create a simple B-spline curve
control_points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
degree = 3
curve = iga.BsplineCurve(control_points, degree)

# Evaluate the curve
t = np.linspace(0, 1, 100)
points = curve.evaluate(t)
```

## Development

### Setting up development environment

1. Clone the repository:
```bash
git clone https://github.com/pantolin/dolfinx_iga.git
cd dolfinx_iga
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate dolfinx_iga
```

3. Install in development mode:
```bash
pip install -e .[dev]
```

3. Run tests:
```bash
pytest
```

### Running tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dolfinx_iga

# Run specific test file
pytest tests/test_bspline.py
```

## Documentation

Documentation is available in the `docs/` directory. To build the documentation:

```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DOLFINx](https://github.com/FEniCS/dolfinx) - The finite element library this extends
- [tIGAr](https://github.com/david-kamensky/tIGAr) - IGA library for legacy FEniCS

## Citation

If you use this software in your research, please cite:

```bibtex
@software{dolfinx_iga,
  title={dolfinx_iga: Isogeometric Analysis Extension for DOLFINx},
  author={Pablo Antolin},
  institution={EPFL (École Polytechnique Fédérale de Lausanne)},
  year={2025},
  url={https://github.com/pantolin/dolfinx_iga}
}
```
