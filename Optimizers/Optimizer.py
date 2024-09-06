from abc import ABC
from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_number_in_range, validate_positive


class Optimizer(ABC):
    def __init__(self,
                 init_learning_rate: float = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0,
                 schedular: Schedular = None):
        if init_learning_rate is not None:
            if validate_positive(init_learning_rate):
                self.learning_rate = init_learning_rate
            else:
                raise ValueError(f"init_learning_rate must be positive float instead got {init_learning_rate}")
        if schedular is None or isinstance(schedular, Schedular):
            self.schedular = schedular
        else:
            raise ValueError(f"schedular must be None or inherit from the Schedular class instead got:{schedular.__class__}")

        if init_learning_rate is None and schedular is None:
            raise ValueError("both schedular and init_learning_rate cant be None, only one of them")

        if validate_number_in_range(n=momentum,
                                    include_lower=True,
                                    include_upper=False):
            self.momentum = momentum
        else:
            raise ValueError(f"momentum must be in range [0, 1), instead got:{momentum}")

        if schedular is None or isinstance(schedular, Schedular):
            self.schedular = schedular
        else:
            raise ValueError(f"schedular must be None or inherit from the Schedular class instead got:{schedular.__class__}")

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
        if self.schedular is not None:
            self.learning_rate = self.schedular.update(*args, **kwargs)
