import math
from Optimizers.Consts import INITIAL_LEARNING_RATE
from Optimizers.Schedules.ExponentialDecay import ExponentialDecay


class InverseTimeDecay(ExponentialDecay):
    def __init__(self,
                 learning_rate: float = INITIAL_LEARNING_RATE,
                 decay_steps: int = 1000,
                 decay_rate: float = 0.1,
                 staircase: bool = False):
        super().__init__(learning_rate, decay_steps, decay_rate, staircase)

    def update(self, *args, **kwargs):
        """
        Apply update to the optimizer learning rate
        """
        self.curr_step += 1
        rate = self.curr_step / self.decay_steps
        if self.staircase:
            rate = math.floor(rate)

        self.learning_rate = self.init_learning_rate / (1 + self.decay_rate * rate)
        return self.learning_rate
