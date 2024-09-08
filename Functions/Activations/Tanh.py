from Functions.Activations.Activation import Activation
import cupy as cp


class Tanh(Activation):
    def activate(self, x):
        """
         Calculate Hyperbolic tangent: (exp(x) - exp(-x))/(exp(x) + exp(-x))
         default value for a is 0.01
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        xp = cp.get_array_module(x)
        return xp.tanh(x)

    def derivative(self, x, **kwargs):
        """
        Calculate Hyperbolic tangent derivative: 1 - ((exp(x) - exp(-x))/(exp(x) + exp(-x)))^2
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the first derivative applied using the activation function on point X
        """
        tanh_x = self.activate(x)
        return 1 - tanh_x ** 2
