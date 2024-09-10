from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Optimizer import Optimizer
import cupy as cp

from System.Utils.Validations import validate_positive


class Adagrad(Optimizer):
    def __init__(self,
                 init_learning_rate: float = INITIAL_LEARNING_RATE,
                 momentum: float = 0.0,
                 epsilon=1e-6,):
        super().__init__(init_learning_rate=init_learning_rate,
                         momentum=momentum)
        if validate_positive(epsilon):
            self.epsilon = epsilon
        else:
            raise ValueError(f"epsilon must be positive small float, recommended to be very small,"
                             f"instead got {epsilon}")

    def apply_gradients(self, gradients, variables, velocity=0, regularizer=0):
        velocity = self.get_velocity_of_momentum(gradients, velocity)
        xp = cp.get_array_module(gradients)
        fraction = 1 / (xp.sqrt(xp.sum(gradients ** 2)) + self.epsilon)
        return (variables - self.get_learning_rate() * fraction * (velocity + regularizer)), velocity
