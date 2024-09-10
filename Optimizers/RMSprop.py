from typing import Union
import cupy as cp
from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Optimizer import Optimizer
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_positive, validate_number_in_range


class RMSprop(Optimizer):
    def __init__(self,
                 init_learning_rate: Union[float, Schedular] = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0,
                 epsilon=1e-6,
                 rmsprop_momentum: float = 0.99):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum)
        if validate_positive(epsilon):
            self.epsilon = epsilon
        else:
            raise ValueError(f"epsilon must be positive small float, recommended to be very small,"
                             f"instead got {epsilon}")
        if validate_number_in_range(n=rmsprop_momentum,
                                    include_lower=True,
                                    include_upper=False):
            self.rmsprop_momentum = rmsprop_momentum
        else:
            raise ValueError(f"rmsprop_momentum must be in range [0, 1), instead got:{rmsprop_momentum}")

        self.rmsprop_velocities_dict = {}

    def apply_gradients(self, gradients, variables, velocity=0, regularizer=0):
        xp = cp.get_array_module(variables)
        rmsprop_velocity = self._get_rmsprop_velocity(variables)
        rmsprop_velocity = (self.rmsprop_momentum * rmsprop_velocity) + ((1 - self.rmsprop_momentum) * xp.sum(gradients**2))
        self._set_rmsprop_velocity(variables, rmsprop_velocity)
        normalized_rmsprop_velocity = 1 / (xp.sqrt(rmsprop_velocity) + self.epsilon)
        if self.momentum > 0:
            velocity = self.get_velocity_of_momentum(gradients, velocity)
        else:
            velocity = gradients
        updated_variables = variables - self.get_learning_rate() * normalized_rmsprop_velocity * (velocity + regularizer)
        return updated_variables, velocity

    def _get_rmsprop_velocity(self, variables):
        # Use the id of the weight matrix to identify its momentum
        xp = cp.get_array_module(variables)
        variables_id = id(variables)
        if variables_id not in self.rmsprop_velocities_dict:
            # Initialize momentum if not already present
            self.rmsprop_velocities_dict[variables_id] = 0
        return self.rmsprop_velocities_dict[variables_id]

    def _set_rmsprop_velocity(self, variables, velocity):
        self.rmsprop_velocities_dict[id(variables)] = velocity
