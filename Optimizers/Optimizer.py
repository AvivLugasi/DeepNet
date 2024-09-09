from abc import ABC
from typing import Union

from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_number_in_range, validate_positive


class Optimizer(ABC):
    def __init__(self,
                 init_learning_rate: Union[float, Schedular] = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0):
        if init_learning_rate is not None:
            if (isinstance(init_learning_rate, float) and validate_positive(init_learning_rate)) \
                    or isinstance(init_learning_rate, Schedular):
                self.learning_rate = init_learning_rate
            else:
                raise ValueError(f"init_learning_rate must be positive float or "
                                 f"Schedular instead got {init_learning_rate}")
        else:
            raise ValueError("init_learning_rate cant be None")

        if validate_number_in_range(n=momentum,
                                    include_lower=True,
                                    include_upper=False):
            self.momentum = momentum
        else:
            raise ValueError(f"momentum must be in range [0, 1), instead got:{momentum}")

    def apply_gradients(self, *args, **kwargs):
        """
        Apply gradients to variables. This method needs to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_config(self):
        """
        Return configuration details for the optimizer.
        """
        return {"learning_rate": self.learning_rate}

    def update_learning_rate(self, *args, **kwargs):
        if isinstance(self.learning_rate, Schedular):
            self.learning_rate.update(*args, **kwargs)

    def get_velocity(self, gradients, regularizer=0, velocity=0):
        return self.momentum * velocity + (1 - self.momentum) * (gradients + regularizer)

    def get_learning_rate(self):
        if callable(self.learning_rate):
            lr = self.learning_rate()  # Call it if it's a function or callable object
        else:
            lr = self.learning_rate  # Use it directly if it's a float
        return lr
