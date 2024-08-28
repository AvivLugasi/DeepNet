from abc import ABC, abstractmethod
from typing import Union, Tuple
import cupy as cp
from Initializers.Consts import CONSTANT_DEFAULT_SHAPE


class Initializer(ABC):
    @abstractmethod
    def initialize(self, xp=cp):
        pass

    @abstractmethod
    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        pass
