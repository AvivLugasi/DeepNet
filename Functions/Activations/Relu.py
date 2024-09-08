from Functions.Activations.Activation import Activation
import cupy as cp


class Relu(Activation):
    def activate(self, x):
        """
         Calculate max(x,0)
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        xp = cp.get_array_module(x)
        return xp.maximum(x, 0)

    def derivative(self, x, **kwargs):
        """
        return 1 when x > 0, return 0 otherwise
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the first derivative applied using the activation function on point X
        """
        xp = cp.get_array_module(x)
        return xp.where(x > 0, 1, 0)
