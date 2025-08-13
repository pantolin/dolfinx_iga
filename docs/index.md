# dolfinx_iga Documentation

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
api_reference
examples
contributing
```

**dolfinx_iga** is a Python package for Isogeometric Analysis (IGA) that extends DOLFINx with B-spline and NURBS functionality.

## Overview

This library provides:
- B-spline curves and surfaces with Cox-de Boor algorithms
- NURBS (rational B-splines) for exact geometry representation  
- Optional Numba acceleration for performance
- Type hints and comprehensive documentation
- Integration ready for DOLFINx finite element framework

## Quick Start

```python
import dolfinx_iga as iga
import numpy as np

# Create a simple B-spline curve
control_points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
degree = 3
curve = iga.BSplineCurve(control_points, degree)

# Evaluate the curve
t = np.linspace(0, 1, 100)
points = curve.evaluate(t)
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
