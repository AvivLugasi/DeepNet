from typing import Literal
from Optimizers.Consts import OPTIMIZERS_VALID_FUNCTIONS, INITIAL_LEARNING_RATE
from Optimizers.RMSprop import RMSprop
from Optimizers.SGD import SGD


def return_optimizer_from_str(optimizer_name: Literal[OPTIMIZERS_VALID_FUNCTIONS]):
    if optimizer_name not in OPTIMIZERS_VALID_FUNCTIONS:
        raise ValueError(f"func_name must be one of {OPTIMIZERS_VALID_FUNCTIONS}, got {optimizer_name} instead")
    else:
        if optimizer_name == OPTIMIZERS_VALID_FUNCTIONS[0]:
            return SGD(init_learning_rate=INITIAL_LEARNING_RATE)
        elif optimizer_name == OPTIMIZERS_VALID_FUNCTIONS[1]:
            return RMSprop(init_learning_rate=INITIAL_LEARNING_RATE)
