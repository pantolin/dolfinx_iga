"""
dolfinx_iga: Isogeometric Analysis extension for DOLFINx

A modern IGA library built for the FEniCSx ecosystem, providing B-spline
and NURBS functionalities with integration to DOLFINx.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "__version__",
]
