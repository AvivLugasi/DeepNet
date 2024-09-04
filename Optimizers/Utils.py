from typing import Literal

from Optimizers.Consts import OPTIMIZERS_VALID_FUNCTIONS
from Optimizers.Optimizer import Optimizer
from Optimizers.SGD import SGD


def return_optimizer_from_str(optimizer_name: Literal[OPTIMIZERS_VALID_FUNCTIONS]) -> Optimizer():
    if optimizer_name not in OPTIMIZERS_VALID_FUNCTIONS:
        raise ValueError(f"func_name must be one of {OPTIMIZERS_VALID_FUNCTIONS}, got {optimizer_name} instead")
    else:
        if optimizer_name == OPTIMIZERS_VALID_FUNCTIONS[0]:
            return SGD()
