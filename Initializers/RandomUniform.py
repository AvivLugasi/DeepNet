from typing import Union, Tuple
import cupy as cp
from Initializers.Initializer import Initializer
from System.Utils.Validations import validate_xp_module
from Initializers.Consts import CONSTANT_DEFAULT_SHAPE, UNIFORM_HIGH_DEFAULT_VALUE, UNIFORM_LOW_DEFAULT_VALUE


def _validate_low_smaller_than_high(low: float, high: float):
    """
    Ensure low is not greater than high
    """
    if low > high:
        raise ValueError("Parameter low must be smaller or equall to high")


class RandomUniform(Initializer):
    def __init__(self,
                 low: float = UNIFORM_LOW_DEFAULT_VALUE,
                 high: float = UNIFORM_HIGH_DEFAULT_VALUE,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _validate_low_smaller_than_high(low=low,
                                        high=high)
        self.low = low
        self.high = high
        self.shape = shape

    def initialize(self, xp=cp):
        validate_xp_module(xp)
        return xp.random.uniform(low=self.low,
                                 high=self.high,
                                 size=self.shape)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.shape = shape
