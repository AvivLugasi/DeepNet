from Functions.Activations.Activation import Activation
import cupy as cp
import numpy as np
from scipy.special import softmax as cpu_softmax
from cupyx.scipy.special import softmax as gpu_softmax


class Softmax(Activation):
    def activate(self, x):
        """
         Calculate the softmax value of each col of x: exp(x_i)/sum(exp(x_i))
         where i run from 1 to number of cols
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        xp = cp.get_array_module(x)
        if xp == np:
            return cpu_softmax(x, axis=0)
        else:
            return gpu_softmax(x, axis=0)

    def derivative(self, x):
        xp = cp.get_array_module(x)
        # change and think about a method to calculate both ways/ limit softmax to be used only on last layer? combined with cross entropy?
        return 1
