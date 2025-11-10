"""This module provides a Bspline1D class that handles B-spline basis functions
with support for open, floating, and periodic knot vectors.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from dolfinx_iga.splines.bspline_1D import Bspline1D


class Bspline:
    """A class representing a multi-dimensional B-spline made up of 1D splines."""

    def __init__(self, splines: list[Bspline1D]):
        """Initialize a multi-dimensional B-spline.

        Args:
            splines (list[Bspline1D]): The 1D splines that define the multi-dimensional B-spline space.
        """
        self._splines = splines

        self._check_input_validity()

    def _check_input_validity(self) -> None:
        """Check if the input splines are valid.

        The splines must have at least one dimension and all splines must have the same dtype.

        Raises:
            ValueError: If the splines are invalid.
        """
        if self.dim == 0:
            raise ValueError("Splines must have at least one dimension")

        dtype = self._splines[0].dtype
        if not all(spline.dtype == dtype for spline in self._splines):
            raise ValueError("All splines must have the same dtype")

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the splines.

        Returns:
            int: The parametric dimension of the splines.
        """
        return len(self._splines)

    @property
    def splines_1D(self) -> list[Bspline1D]:
        """Get the 1D splines that define the multi-dimensional B-spline space.

        Returns:
            list[Bspline1D]: The 1D splines that define the multi-dimensional B-spline space.
        """
        return self._splines

    @property
    def degrees(self) -> list[int]:
        """Get the degrees of the 1D splines along the parametric dimensions.

        Returns:
            list[int]: The degrees of the 1D splines along the parametric dimensions.
        """
        return [spline.degree for spline in self._splines]

    @property
    def dtype(self) -> np.dtype:
        """Get the data type used in the splines.

        Returns:
            np.dtype: The data type used in the splines.
        """
        return self._splines[0].dtype
