from typing import Union
import cupy as cp
from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.RMSprop import RMSprop
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_positive, validate_number_in_range


class Adam(RMSprop):
    def __init__(self,
                 init_learning_rate: Union[float, Schedular] = INITIAL_LEARNING_RATE,
                 momentum: float = 0.99,
                 epsilon=1e-6,
                 optimizer_momentum: float = 0.9):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum,
                         epsilon=epsilon,
                         optimizer_momentum=optimizer_momentum)

    def apply_gradients(self, gradients, variables, velocity=0, regularizer=0):
        xp = cp.get_array_module(variables)
        optimizer_velocity, iteration = self._get_optimizer_variable_velocity(variables)
        optimizer_velocity = (self.optimizer_momentum * optimizer_velocity) + (
                    (1 - self.optimizer_momentum) * xp.sum(gradients**2))
        self._set_optimizer_variable_velocity(variables, optimizer_velocity)
        velocity = self.get_velocity_of_momentum(gradients, velocity)
        normalized_optimizer_velocity = optimizer_velocity / (1 - self.optimizer_momentum**(iteration+1))
        normalized_velocity = velocity / (1 - self.momentum**(iteration+1))
        fraction = 1 / xp.sqrt(normalized_optimizer_velocity + self.epsilon)
        updated_variables = variables - self.get_learning_rate() * fraction * (normalized_velocity + regularizer)
        return updated_variables, velocity
