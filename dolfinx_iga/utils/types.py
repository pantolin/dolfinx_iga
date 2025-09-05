from typing import Union

import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.floating]
FloatLike = Union[float, list[float], np.floating, FloatArray]

# Use Union instead of | for better compatibility
Float_32_64 = Union[np.float32, np.float64]
FloatArray_32 = npt.NDArray[np.float32]
FloatArray_64 = npt.NDArray[np.float64]
FloatArray_32_64 = npt.NDArray[np.float32] | npt.NDArray[np.float64]
FloatLike_32_64 = Union[
    float, list[float], np.float32, np.float64, FloatArray_32, FloatArray_64
]
FloatLikeArray_32_64 = Union[list[float], list[int], FloatArray_32, FloatArray_64]
IntArray = npt.NDArray[np.int_]
