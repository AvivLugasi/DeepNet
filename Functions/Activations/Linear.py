from Functions.Activations.Activation import Activation


class Linear(Activation):
    def __init__(self,
                 a: float = 1):
        self.a = a

    def activate(self, x):
        """
        Return ax where a is a constant
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the transformation applied using the activation function
        """
        return self.a * x

    def derivative(self, x, **kwargs):
        """
        return a where a is a constant
        :param x: The linear combination of W*X + b of a neuron
        :return: The value of the first derivative applied using the activation function on point X
        """
        return self.a
