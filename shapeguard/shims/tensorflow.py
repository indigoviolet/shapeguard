from typing import List, Optional
from .shim import TensorShim
import tf  # type: ignore


class TfTensorShim(TensorShim[tf.Tensor]):
    def get_shape(self) -> List[int]:
        return self.tensor.get_shape().as_list()

    def reshape(self, new_shape: List[Optional[int]]) -> tf.Tensor:
        return tf.reshape(self.tensor, new_shape)


class TfTensorShapeShim(TensorShim[tf.TensorShape]):
    def __init__(self, tensor: tf.TensorShape):
        self.tensor = tensor

    def get_shape(self) -> List[int]:
        return self.tensor.as_list()
