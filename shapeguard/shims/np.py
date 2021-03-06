from typing import List

import numpy as np

from .shim import TensorShim


class NpTensorShim(TensorShim[np.ndarray]):
    def get_shape(self) -> List[int]:
        return list(self.tensor.shape)
