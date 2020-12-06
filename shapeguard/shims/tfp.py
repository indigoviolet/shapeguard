from typing import List

import tensorflow_probability as tfp

from .shim import TensorShim


class TfpDistributionShim(TensorShim[tfp.distribution.Distribution]):
    def get_shape(self) -> List[int]:
        return self.tensor.batch_shape.as_list() + self.tensor.event_shape.as_list()  # type: ignore
