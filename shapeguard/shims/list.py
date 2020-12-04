from typing import List, Union, Tuple
from .shim import TensorShim

ListShapeType = Union[List[int], Tuple[int, ...]]


class ListTensorShim(TensorShim[ListShapeType]):
    def get_shape(self) -> List[int]:
        return list(self.tensor)
