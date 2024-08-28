from typing import Union, Tuple
import cupy as cp
from Initializers.Initializer import Initializer
from System.Utils.Validations import validate_xp_module
from Initializers.Consts import CONSTANT_DEFAULT_SHAPE


class Identity(Initializer):
    def __init__(self,
                 n: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.n = n if isinstance(n, int) else n[0]

    def initialize(self, xp=cp):
        validate_xp_module(xp)
        return xp.identity(n=self.n)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.n = shape if isinstance(shape, int) else shape[0]
