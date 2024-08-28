from Functions.Activations.Activation import Activation
import cupy as cp
import numpy as np
from scipy.special import expit as cpu_expit
from cupyx.scipy.special import expit as gpu_expit


class Sigmoid(Activation):
    def activate(self, x):
        """
         Calculate the sigmoid value of x: 1/(1+exp(-x))
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        xp = cp.get_array_module(x)
        if xp == np:
            return cpu_expit(x)
        else:
            return gpu_expit(x)

    def derivative(self, x):
        """
        Calculate sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the first derivative applied using the activation function on point X
        """
        sigmoid = self.activate(x)
        return sigmoid * (1 - sigmoid)
