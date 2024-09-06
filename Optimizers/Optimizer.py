from abc import ABC
from Optimizers.Consts import INITIAL_LEARNING_RATE
from System.Utils.Validations import validate_number_in_range, validate_positive


class Optimizer(ABC):
    def __init__(self,
                 init_learning_rate: float = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0):
        if validate_positive(init_learning_rate):
            self.init_learning_rate = init_learning_rate
        else:
            raise ValueError("init_learning_rate must be positive float instead got {init_learning_rate}")

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
        return {"learning_rate": self.init_learning_rate}
