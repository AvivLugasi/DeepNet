from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_positive, validate_positive_int
import math


class ExponentialDecay(Schedular):
    def __init__(self,
                 learning_rate: float = INITIAL_LEARNING_RATE,
                 decay_steps: int = 1000,
                 decay_rate: float = 0.95,
                 staircase: bool = False):
        super().__init__(learning_rate=learning_rate)
        if validate_positive_int(decay_steps) and validate_positive(decay_rate):
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
        else:
            raise ValueError(f"decay_steps and decay_rate must be positive instead got:{decay_steps}, {decay_rate}")
        self.staircase = staircase

    def update(self, *args, **kwargs):
        """
        Apply update to the optimizer learning rate
        """
        self.curr_step += 1

        rate = self.curr_step / self.decay_steps
        if self.staircase:
            rate = math.floor(rate)
        self.learning_rate = self.init_learning_rate * math.pow(self.decay_rate, rate)
        return self.learning_rate
