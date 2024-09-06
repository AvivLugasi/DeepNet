from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_positive, validate_positive_int
import math


class PolynomialDecay(Schedular):
    def __init__(self,
                 learning_rate: float = INITIAL_LEARNING_RATE,
                 decay_steps: int = 100,
                 end_learning_rate: float = 0.0001,
                 power: float = 1,
                 cycle: bool = False
                 ):
        super().__init__(learning_rate=learning_rate)
        if validate_positive_int(decay_steps):
            self.decay_steps = decay_steps
        else:
            raise ValueError(f"decay_steps must be positive int instead got:{decay_steps}")
        if validate_positive(end_learning_rate) and end_learning_rate < self.init_learning_rate:
            self.end_learning_rate = end_learning_rate
        else:
            raise ValueError(
                f"end_learning_rate must be positive and smaller then learning_rate, instead got:{end_learning_rate}")
        if validate_positive(power):
            self.power = power
        else:
            raise ValueError(f"power must be positive instead got:{power}")
        self.cycle = cycle

    def update(self, *args, **kwargs):
        """
        Apply update to the optimizer learning rate
        """
        self.curr_step += 1
        if not self.cycle:
            step = min(self.curr_step, self.decay_steps)
            decay_steps = self.decay_steps
        else:
            step = self.curr_step
            decay_steps = self.decay_steps * math.ceil(self.curr_step / self.decay_steps)
        self.learning_rate = (self.init_learning_rate - self.end_learning_rate) * math.pow((1 - step / decay_steps), self.power)
        return self.learning_rate
