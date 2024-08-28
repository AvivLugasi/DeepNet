from typing import Union, Tuple
import cupy as cp
from Initializers.Initializer import Initializer
from System.Utils.Validations import validate_xp_module
from Initializers.Constants import Constants, CONSTANT_DEFAULT_SHAPE


class Zeroes(Initializer):
    def __init__(self,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE):
        self.initializer = Constants(value=0.0,
                                     shape=shape)

    def initialize(self, xp=cp):
        validate_xp_module(xp)
        return self.initializer.initialize(xp=xp)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.initializer.set_shape(shape=shape)
