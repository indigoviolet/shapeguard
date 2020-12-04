from typing import List
from .shim import TensorShim
import tensorflow_probability as tfp  # type: ignore


class TfpDistributionShim(TensorShim[tfp.distribution.Distribution]):
    def get_shape(self) -> List[int]:
        return self.tensor.batch_shape.as_list() + self.tensor.event_shape.as_list()
