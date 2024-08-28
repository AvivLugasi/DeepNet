from Functions.Activations.Activation import Activation
import cupy as cp


class LeakyRelu(Activation):
    def __init__(self,
                 starting_alpha_value: float = 0.01,
                 is_constant: bool = True):
        # Ensure starting_alpha_value is a positive float
        if starting_alpha_value <= 0:
            raise ValueError("starting_alpha_value must be a positive float.")

        self.__alpha = starting_alpha_value
        self.is_constant = is_constant

    def activate(self, x):
        """
         Return x when x > 0, a*x otherwise, alpha is a learned parameter(if is constant = false)
         default value for a is 0.01
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        xp = cp.get_array_module(x)
        return xp.where(x > 0, x, self.__alpha * x)

    def derivative(self, x):
        """
        return 1 when x > 0, return a otherwise, alpha is a learned parameter(if is constant = false)
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the first derivative applied using the activation function on point X
        """
        xp = cp.get_array_module(x)
        return xp.where(x > 0, 1, self.__alpha)

    def update_alpha(self, loss_gradient):
        if not self.is_constant:
            self.__alpha -= loss_gradient
