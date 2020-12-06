from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")


class TensorShim(Generic[T]):
    def __init__(self, tensor: T):
        self.tensor = tensor

    def get_shape(self) -> List[int]:
        raise NotImplementedError

    def reshape(self, new_shape: List[Optional[int]]) -> T:
        raise NotImplementedError


try:
    from numpy import ndarray as _ndarray
except ImportError:
    ndarray = None
else:
    ndarray = _ndarray

try:
    from tensorflow import Tensor as _TfTensor
    from tensorflow import TensorShape as _TfTensorShape
except ImportError:
    TfTensor, TfTensorShape = None, None
else:
    TfTensor, TfTensorShape = _TfTensor, _TfTensorShape

try:
    import tensorflow_probability.distributions.Distribution as _TfpDistribution
except ImportError:
    TfpDistribution = None
else:
    TfpDistribution = _TfpDistribution

try:
    from torch import Tensor as _TorchTensor
except ImportError:
    TorchTensor = None
else:
    TorchTensor = _TorchTensor


def get_shim(tensor: Any) -> TensorShim:
    if isinstance(tensor, (list, tuple)):
        from .list import ListTensorShim

        return ListTensorShim(tensor)

    elif TfTensor is not None and isinstance(tensor, TfTensor):

        from .tf import TfTensorShim

        return TfTensorShim(tensor)

    elif TfTensorShape is not None and isinstance(tensor, TfTensorShape):

        from .tf import TfTensorShapeShim

        return TfTensorShapeShim(tensor)

        return tensor.as_list()
    elif ndarray is not None and isinstance(tensor, ndarray):

        from .np import NpTensorShim

        return NpTensorShim(tensor)

    elif TfpDistribution is not None and isinstance(tensor, TfpDistribution):

        from .tfp import TfpDistributionShim

        return TfpDistributionShim(tensor)

    elif TorchTensor is not None and isinstance(tensor, TorchTensor):

        from .pytorch import TorchTensorShim

        return TorchTensorShim(tensor)
    else:
        raise TypeError(
            "Unknown tensor/shape {} of type: {}".format(tensor, type(tensor))
        )
