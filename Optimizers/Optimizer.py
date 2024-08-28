from abc import ABC


class Optimizer(ABC):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def apply_gradients(self, gradients, variables):
        """
        Apply gradients to variables. This method needs to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_config(self):
        """
        Return configuration details for the optimizer.
        """
        return {"learning_rate": self.learning_rate}
