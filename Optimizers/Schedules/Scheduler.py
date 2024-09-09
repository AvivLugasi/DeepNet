from abc import ABC
from Optimizers.Consts import INITIAL_LEARNING_RATE
from System.Utils.Validations import validate_positive


class Schedular(ABC):
    def __init__(self, learning_rate: float = INITIAL_LEARNING_RATE):
        self.curr_step = 0
        if validate_positive(learning_rate):
            self.learning_rate = learning_rate
            self.init_learning_rate = learning_rate
        else:
            raise ValueError(f"learning_rate must be positive float instead got {learning_rate}")

    def __call__(self, *args, **kwargs):
        return self.learning_rate

    def update(self, *args, **kwargs):
        """
        Apply update to the optimizer learning rate
        """
        raise NotImplementedError("Subclasses should implement this method.")
