from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self,
                 init_learning_rate: float = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum)

    def apply_gradients(self, gradients, variables, regularizer=0, velocity=0):
        if regularizer is None:
            regularizer = 0
        velocity = self.momentum * velocity + (1 - self.momentum) * (gradients + regularizer)
        return (variables - self.init_learning_rate * velocity), velocity
