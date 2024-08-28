from typing import Union
import numpy as np
import cupy as cp
from Structures.Layers.Layer import Layer


class Dense(Layer):
    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        pass

    def backward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        pass
