import numpy as np

from .utils.tolerance import get_default_tolerance, unique_with_tolerance


class KnotsVector:
    def __init__(self, knots_with_repetitions: np.typing.NDArray[np.floating]) -> None:
        self._knots_with_repetitions = knots_with_repetitions
        self._unique_knots = np.unique(self._knots_with_repetitions)

    def __len__(self) -> int:
        return len(self._knots_with_repetitions)

    def __getitem__(self, index) -> np.floating:
        return self._knots_with_repetitions[index]

    def __repr__(self) -> str:
        return f"KnotsVector({self._knots_with_repetitions.tolist()})\n{self.__str__()}"

    def __str__(self) -> str:
        knots_str = np.array2string(
            self._knots_with_repetitions, precision=4, suppress_small=True
        )
        unique_str = np.array2string(
            self._unique_knots, precision=4, suppress_small=True
        )
        return f"Knots with repetitions: {knots_str}\nUnique knots: {unique_str}"

    @property
    def knots_with_repetitions(self) -> np.typing.NDArray[np.floating]:
        """Get the knot vector with repetitions."""
        return self._knots_with_repetitions

    @property
    def unique_knots(self) -> np.typing.NDArray[np.floating]:
        """Get the unique knot values."""
        return self._unique_knots

    @property
    def dtype(self) -> np.dtype:
        """Get the floating point data type of the knot vector."""
        return self._knots_with_repetitions.dtype

    def multiplicities(self, tol: np.floating | None = None) -> dict[np.floating, int]:
        """Get the multiplicity of each unique knot value.

        Args:
            tol: Tolerance for considering knots as equal. If None, uses a default
                 based on the dtype precision.
        """
        if tol is None:
            tol = get_default_tolerance(self.dtype)

        # Use the tolerance utility function
        unique, counts = unique_with_tolerance(
            self._knots_with_repetitions, custom_tolerance=tol
        )
        return dict(zip(unique, counts))
