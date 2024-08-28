from typing import Union, Tuple
import cupy as cp
from Initializers.Initializer import Initializer
from System.Utils.Validations import validate_xp_module
from Initializers.Consts import CONSTANT_DEFAULT_SHAPE, NORMAL_MEAN_DEFAULT_VALUE, NORMAL_STD_DEFAULT_VALUE


class RandomNormal(Initializer):
    def __init__(self,
                 std: float = NORMAL_STD_DEFAULT_VALUE,
                 mean: float = NORMAL_MEAN_DEFAULT_VALUE,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.mean = mean
        self.std = std
        self.shape = shape

    def initialize(self, xp=cp):
        validate_xp_module(xp)
        # Ensure shape is a tuple
        if isinstance(self.shape, int):
            dimensions = (self.shape,)
        else:
            dimensions = self.shape
        return self.mean + self.std * xp.random.randn(*dimensions)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.shape = shape
