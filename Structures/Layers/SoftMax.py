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

    def backward_pass(self, **kwargs):
        grads = kwargs.get('grads')  # Gradient from the next layer, shape: (num_classes, batch_size)
        xp = cp.get_array_module(grads)

        # Output is of shape (num_classes, batch_size)
        num_classes = self.output.shape[0]
        batch_size = self.output.shape[1]

        # Initialize the gradient with respect to the input, same shape as the output
        d_x = xp.empty_like(self.output)

        for i in range(batch_size):
            # Get the softmax output for the i-th sample (shape: (num_classes,))
            softmax_out = self.output[:, i].reshape(-1, 1)  # Shape: (num_classes, 1)

            # Compute the Jacobian matrix for the i-th sample
            jacobian = xp.diagflat(softmax_out) - xp.dot(softmax_out, softmax_out.T)

            # Compute the gradient for this sample (grads[:, i] is shape (num_classes,))
            d_x[:, i] = xp.dot(jacobian, grads[:, i])

        return d_x

    def update_weights(self, **kwargs):
        pass
