from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_positive_int, validate_number_in_range
import cupy as cp


class CosineDecay(Schedular):
    def __init__(self,
                 learning_rate: float = INITIAL_LEARNING_RATE,
                 decay_steps: int = 1000,
                 alpha: float = 0.0):
        super().__init__(learning_rate=learning_rate)
        if validate_positive_int(decay_steps):
            self.decay_steps = decay_steps
        else:
            raise ValueError(f"decay_steps must be positive int instead got:{decay_steps}")
        if validate_number_in_range(n=alpha, include_upper=False):
            self.alpha = alpha
        else:
            raise ValueError(f"alpha must be in range [0,1), instead got:{alpha}")

    def update(self, *args, **kwargs):
        """
        Apply update to the optimizer learning rate
        """
        self.curr_step += 1
        self.curr_step = min(self.decay_steps, self.curr_step)
        cosine_decay = 0.5 * self.init_learning_rate * (1 + cp.cos((cp.pi * self.curr_step) / self.decay_steps))
        self.learning_rate = ((1 - self.alpha) * cosine_decay + self.alpha) * self.init_learning_rate
        return self.learning_rate
