from typing import List
from .shim import TensorShim
import torch  # type: ignore


class TorchTensorShim(TensorShim[torch.Tensor]):
    def get_shape(self) -> List[int]:
        return self.tensor.shape()
