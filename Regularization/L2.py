from typing import Union
import numpy as np
import cupy as cp
from Regularization.Regularizer import Regularizer
from System.Utils.Validations import validate_positive


class L2(Regularizer):
    def __init__(self, l2: float = 0.01):
        if validate_positive(l2):
            self.l2 = l2
        else:
            raise ValueError(f"l2 must be positive, instead got {l2}")

    def __call__(self, x: Union[np.ndarray, cp.ndarray]):
        """
        Apply L2 regularization on the given parameter tensor.
        Computed as:

        :param x: Union[np.ndarray, cp.ndarray] - layer parameters tensor to apply the regularization on.
        :return: The regularization penalty derivative
        """
        return self.l2 * x

    def cost(self, x: Union[np.ndarray, cp.ndarray]):
        xp = cp.get_array_module(x)
        return self.l2 * xp.sum(x**2)

    def get_config(self):
        """
        Return the regularizer config
        """
        return f"Regularizer: L2, L2 value: {self.l2}"
