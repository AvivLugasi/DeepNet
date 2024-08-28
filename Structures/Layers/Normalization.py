from typing import Literal
import cupy as cp
import numpy as np
from Structures.Layers.Layer import Layer
from PreProcessing.Normalization import *


def _return_normalization_func_from_str(func_name: Literal["standardization", "minmax_scalar"] = "standardization"):
    if func_name == "standardization":
        return standardization
    elif func_name == "minmax_scalar":
        return minmax_scalar
    else:
        raise ValueError(f"func_name value: {func_name} is invalid")


class Normalization(Layer):
    def __init__(self, method: Literal["standardization", "minmax_scalar"] = "standardization"):
        self.normalization_func = _return_normalization_func_from_str(func_name=method)

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        return self.normalization_func(input_mat=input_mat)

    def backward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        return input_mat
