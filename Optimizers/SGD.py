from Optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, init_learning_rate: float = None):
        super().__init__(init_learning_rate=init_learning_rate)

    def apply_gradients(self, gradients, variables, regularizer=0):
        if regularizer is None:
            regularizer = 0

        return variables - self.init_learning_rate * gradients - self.init_learning_rate * regularizer
