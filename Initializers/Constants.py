from typing import Union, Tuple
import cupy as cp
from Initializers.Initializer import Initializer
from System.Utils.Validations import validate_xp_module
from Initializers.Consts import CONSTANT_DEFAULT_VALUE, CONSTANT_DEFAULT_SHAPE


class Constants(Initializer):
    def __init__(self,
                 value: float = CONSTANT_DEFAULT_VALUE,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE):
        self.value = value
        self.shape = shape

    def initialize(self, xp=cp):
        validate_xp_module(xp)
        return xp.full(shape=self.shape,
                       fill_value=self.value)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.shape = shape
