from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Optimizer import Optimizer
from Optimizers.Schedules.Scheduler import Schedular


class SGD(Optimizer):
    def __init__(self,
                 init_learning_rate: float = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0,
                 schedular: Schedular = None):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum,
                         schedular=schedular)

    def apply_gradients(self, gradients, variables, regularizer=0, velocity=0):
        velocity = self.momentum * velocity + (1 - self.momentum) * (gradients + regularizer)
        return (variables - self.learning_rate * velocity), velocity
