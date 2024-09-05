from typing import Literal
from Functions.Losses.BinaryCrossEntropy import BinaryCrossEntropy
from Functions.Losses.Consts import LOSS_FUNCTIONS_VALID_VALUES
from Functions.Losses.CrossEntropy import CrossEntropy
from Functions.Losses.Loss import Loss
from Functions.Losses.MSE import MSE


def return_loss_func_from_str(func_name: Literal[LOSS_FUNCTIONS_VALID_VALUES]) -> Loss:
    if func_name not in LOSS_FUNCTIONS_VALID_VALUES:
        raise ValueError(f"func_name must be one of {LOSS_FUNCTIONS_VALID_VALUES}, got {func_name} instead")
    else:
        if func_name == LOSS_FUNCTIONS_VALID_VALUES[0]:
            return MSE()
        elif func_name == LOSS_FUNCTIONS_VALID_VALUES[1]:
            return BinaryCrossEntropy()
        else:
            return CrossEntropy()
