from typing import List

import torch

from .shim import TensorShim


class TorchTensorShim(TensorShim[torch.Tensor]):
    def get_shape(self) -> List[int]:
        return list(self.tensor.shape)
