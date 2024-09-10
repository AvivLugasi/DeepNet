from Functions.Activations.Activation import Activation
import cupy as cp


class Elu(Activation):
    def __init__(self,
                 alpha_value: float = 0.01):
        # Ensure starting_alpha_value is a positive float
        if alpha_value <= 0:
            raise ValueError("alpha_value must be a positive float.")

        self.__alpha = alpha_value

    def activate(self, x):
        """
         Return x when x > 0, a * (exp(x) - 1) otherwise, alpha is a learned parameter(if is constant = false)
         default value for a is 0.01
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        xp = cp.get_array_module(x)
        return xp.where(x > 0, x, self.__alpha * (xp.exp(x) - 1))

    def derivative(self, x, **kwargs):
        """
        return 1 when x > 0, return a * exp(e) otherwise, alpha is a learned parameter(if is constant = false)
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the first derivative applied using the activation function on point X
        """
        xp = cp.get_array_module(x)
        d_x = xp.where(x > 0, 1, self.__alpha * xp.exp(x))
        return d_x
