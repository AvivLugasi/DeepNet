from typing import Union
from sympy.physics.units import momentum
from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Optimizer import Optimizer
from Optimizers.Schedules.Scheduler import Schedular


class RMSprop(Optimizer):
    def __init__(self,
                 init_learning_rate: Union[float, Schedular] = INITIAL_LEARNING_RATE):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum)

    def apply_gradients(self, gradients, variables, regularizer=0, velocity=0):
        pass
