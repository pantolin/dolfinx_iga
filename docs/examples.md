# Examples

## Basic B-spline Curve

```python
import numpy as np
from dolfinx_iga import BSplineCurve

# Define control points
control_points = np.array([
    [0.0, 0.0],
    [1.0, 2.0], 
    [3.0, 1.0],
    [4.0, 0.0]
])

# Create B-spline curve
curve = BSplineCurve(control_points, degree=3)

# Evaluate at parameter values
u_vals = np.linspace(0, 1, 100)
points = curve.evaluate(u_vals)

# Compute derivatives
derivatives = curve.derivative(u_vals)
```

## NURBS Circle Arc

```python
import numpy as np
from dolfinx_iga import NURBSCurve

# Quarter circle arc from (1,0) to (0,1)
control_points = np.array([
    [1.0, 0.0],
    [1.0, 1.0], 
    [0.0, 1.0]
])
weights = np.array([1.0, 1.0/np.sqrt(2), 1.0])

# Create NURBS curve
circle_arc = NURBSCurve(control_points, weights, degree=2)

# Evaluate
u_vals = np.linspace(0, 1, 50)
points = circle_arc.evaluate(u_vals)
```

## B-spline Surface

```python
import numpy as np
from dolfinx_iga import BSplineSurface

# 3x3 control point grid
control_points = np.array([
    [[0, 0, 0], [0, 1, 1], [0, 2, 0]],
    [[1, 0, 1], [1, 1, 2], [1, 2, 1]],
    [[2, 0, 0], [2, 1, 1], [2, 2, 0]]
])

# Create surface
surface = BSplineSurface(control_points, degree_u=2, degree_v=2)

# Evaluate on parameter grid
u_vals = np.linspace(0, 1, 10)
v_vals = np.linspace(0, 1, 10)
points = surface.evaluate(u_vals, v_vals)
```

## More Examples

See the `examples/` directory in the repository for complete, runnable examples:
- `basic_bspline_curve.py` - Complete curve example with plotting
- Additional examples coming soon!
