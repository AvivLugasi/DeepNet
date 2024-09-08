from typing import Union
from Structures.Layers.Consts import CROSS_ENTROPY_AFTER_SOFTMAX_FUNC
from Structures.Layers.Layer import Layer
import numpy as np
import cupy as cp
from scipy.special import softmax as cpu_softmax
from cupyx.scipy.special import softmax as gpu_softmax

from System.Utils.Validations import validate_np_cp_array


def softmax(x: Union[np.ndarray, cp.ndarray],
            axis: int = 0):
    xp = cp.get_array_module(x)
    if xp == np:
        return cpu_softmax(x, axis=axis)
    else:
        return gpu_softmax(x, axis=axis)


class SoftMax(Layer):
    def __init__(self):
        self.output = None

    def forward_pass(self,
                     x: Union[np.ndarray, cp.ndarray],
                     axis: int = 0):
        """
         Calculate the softmax value of each col of x: exp(x_i)/sum(exp(x_i))
         where i run from 1 to number of cols
        :param axis:
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        validate_np_cp_array(x)
        self.output = softmax(x, axis=axis)
        return self.output

    def backward_pass(self, grads: Union[np.ndarray, cp.ndarray]):
        xp = cp.get_array_module(grads)
        n = self.output.shape[0]  # neurons / num of classes are the tensor raws
        return xp.dot((xp.identity(n=n) - self.output.T) * self.output, grads)

    def update_weights(self, **kwargs):
        pass
