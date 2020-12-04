from typing import List
from .shim import TensorShim
import numpy as np  # type: ignore


class NpTensorShim(TensorShim[np.ndarray]):
    def get_shape(self) -> List[int]:
        return list(self.tensor.shape)
