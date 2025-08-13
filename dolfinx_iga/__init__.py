"""
dolfinx_iga: Isogeometric Analysis extension for DOLFINx

A modern IGA library built for the FEniCSx ecosystem, providing B-spline
and NURBS functionalities with integration to DOLFINx.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .bspline import BSplineCurve, BSplineSurface
from .nurbs import NURBSCurve, NURBSSurface
from .utils import basis_functions, knot_vector_utils

__all__ = [
    "__version__",
    "BSplineCurve",
    "BSplineSurface",
    "NURBSCurve",
    "NURBSSurface",
    "knot_vector_utils",
    "basis_functions",
]
