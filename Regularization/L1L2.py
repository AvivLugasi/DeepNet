from typing import Union
import numpy as np
import cupy as cp

from Regularization.L1 import L1
from Regularization.L2 import L2
from Regularization.Regularizer import Regularizer
from System.Utils.Validations import validate_positive


class L1L2(Regularizer):
    def __init__(self,
                 l1: float = 0.01,
                 l2: float = 0.01):
        # Initialize L1 and L2 components
        self.l1 = L1(l1)
        self.l2 = L2(l2)

    def __call__(self, x: Union[np.ndarray, cp.ndarray]):
        """
        Apply L1L2 regularization on the given parameter tensor.

        :param x: Union[np.ndarray, cp.ndarray] - layer parameters tensor to apply the regularization on.
        :return: The regularization penalty derivative
        """
        # Apply both L1 and L2 regularizations
        return self.l1(x) + self.l2(x)

    def get_config(self):
        """
        Return the regularizer config
        """
        return {
            "Regularizer": "L1L2",
            "L1 value": self.l1.l1_value,  # Assuming L1 stores its value
            "L2 value": self.l2.l2_value  # Assuming L2 stores its value
        }
