from abc import ABC
from typing import Union
import numpy as np
import cupy as cp


class Regularizer(ABC):
    def __call__(self, x: Union[np.ndarray, cp.ndarray]):
        """
        Apply regularization on the given parameter tensor
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_config(self):
        """
        Return the regularizer config
        """
        pass
