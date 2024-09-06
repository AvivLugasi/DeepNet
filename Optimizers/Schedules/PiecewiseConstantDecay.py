from typing import List
from Optimizers.Schedules.Scheduler import Schedular
from System.Utils.Validations import validate_positive_increasing_integer_list, \
    validate_positive_decreasing_integer_list


class PiecewiseConstantDecay(Schedular):
    def __init__(self,
                 boundaries: List[int] = None,
                 values: List[float] = None):
        super().__init__()
        if validate_positive_increasing_integer_list(boundaries) and \
                validate_positive_decreasing_integer_list(values) and \
                len(boundaries) == len(values):
            self.boundaries = boundaries
            self.values = values
        else:
            raise ValueError("boundaries must be at the same size of values, values must be positive decreasing and boundaries positive increasing")
        self.index = 0
        self.learning_rate = self.values[0]

    def update(self, *args, **kwargs):
        """
        Apply update to the optimizer learning rate
        """
        self.curr_step += 1
        if self.index < len(self.boundaries) - 1:
            if self.curr_step > self.boundaries[self.index]:
                self.index += 1
                self.learning_rate = self.values[self.index]

        return self.learning_rate
