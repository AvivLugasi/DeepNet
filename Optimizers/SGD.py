from typing import Union

from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Optimizer import Optimizer
from Optimizers.Schedules.Scheduler import Schedular


class SGD(Optimizer):
    def __init__(self,
                 init_learning_rate: Union[float, Schedular] = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum)

    def apply_gradients(self, gradients, variables, regularizer=0, velocity=0):
        velocity = self.get_velocity(gradients, regularizer, velocity)
        return (variables - self.get_learning_rate() * velocity), velocity
