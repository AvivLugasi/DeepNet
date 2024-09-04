from abc import ABC
from Optimizers.Consts import INITIAL_LEARNING_RATE


class Optimizer(ABC):
    def __init__(self, init_learning_rate: float = INITIAL_LEARNING_RATE):
        self.init_learning_rate = init_learning_rate

    def apply_gradients(self, gradients, variables, regularizer):
        """
        Apply gradients to variables. This method needs to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_config(self):
        """
        Return configuration details for the optimizer.
        """
        return {"learning_rate": self.init_learning_rate}
