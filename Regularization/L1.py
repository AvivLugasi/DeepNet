from typing import Union
import numpy as np
import cupy as cp
from Regularization.Regularizer import Regularizer
from System.Utils.Validations import validate_positive


class L1(Regularizer):
    def __init__(self, l1: float = 0.01):
        if validate_positive(l1):
            self.l1 = l1
        else:
            raise ValueError(f"l1 must be positive, instead got {l1}")

    def __call__(self, x: Union[np.ndarray, cp.ndarray]):
        xp = cp.get_array_module(x)
        d_l1 = xp.where(x > 0, self.l1, -self.l1)
        return d_l1

    def cost(self, x: Union[np.ndarray, cp.ndarray]):
        xp = cp.get_array_module(x)
        return self.l1 * xp.sum(xp.abs(x))

    def get_config(self):
        """
        Return the regularizer config
        """
        return f"Regularizer: L1, L1 value: {self.l1}"
