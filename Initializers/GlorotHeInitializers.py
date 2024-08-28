from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
import cupy as cp
from Initializers.Consts import CONSTANT_DEFAULT_SHAPE
from Initializers.Initializer import Initializer
from Initializers.RandomNormal import RandomNormal
from Initializers.RandomUniform import RandomUniform


class _HeGlorotBaseInitializer(Initializer, ABC):
    def __init__(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        self.denominator = self.calc_denominator()  # Ensure this is calculated first
        self.fraction = self.calc_fraction()  # Ensure this is calculated next
        self.initializer = self.initialize_initializer()  # Then initialize

    def _std_range_calculation(self) -> float:
        return np.sqrt(self.fraction / self.denominator)

    @abstractmethod
    def calc_fraction(self) -> float:
        pass

    @abstractmethod
    def calc_denominator(self) -> float:
        pass

    @abstractmethod
    def initialize_initializer(self) -> Initializer:
        pass

    def initialize(self, xp=cp):
        return self.initializer.initialize(xp=xp)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        self.initializer.set_shape(shape=shape)


class _UniformInitializer(ABC):
    def initialize_initializer(self):
        uniform_range = self._std_range_calculation()
        return RandomUniform(low=-uniform_range, high=uniform_range, shape=self.shape)

    def calc_fraction(self):
        return 6


class _NormalInitializer(ABC):
    def initialize_initializer(self):
        std = self._std_range_calculation()
        return RandomNormal(shape=self.shape, std=std)

    def calc_fraction(self):
        return 2


class _GlorotBaseInitializer(_HeGlorotBaseInitializer, ABC):
    def calc_denominator(self):
        return self.shape[1] + self.shape[0]


class _HeBaseInitializer(_HeGlorotBaseInitializer, ABC):
    def calc_denominator(self):
        return self.shape[1]


class GlorotNormal(_GlorotBaseInitializer, _NormalInitializer):
    def __init__(self,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _GlorotBaseInitializer.__init__(self, shape=shape)
        _NormalInitializer.__init__(self)

    # `HeNormal` is now concrete and fully instantiated
    def calc_fraction(self):
        return _NormalInitializer.calc_fraction(self)

    def initialize_initializer(self):
        return _NormalInitializer.initialize_initializer(self)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _GlorotBaseInitializer.set_shape(self, shape=shape)


class GlorotUniform(_GlorotBaseInitializer, _UniformInitializer):
    def __init__(self,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _GlorotBaseInitializer.__init__(self, shape=shape)
        _UniformInitializer.__init__(self)

    # `HeUniform` is now concrete and fully instantiated
    def calc_fraction(self):
        return _UniformInitializer.calc_fraction(self)

    def initialize_initializer(self):
        return _UniformInitializer.initialize_initializer(self)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _GlorotBaseInitializer.set_shape(self, shape=shape)


class HeNormal(_HeBaseInitializer, _NormalInitializer):
    def __init__(self,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _HeBaseInitializer.__init__(self, shape=shape)
        _NormalInitializer.__init__(self)

    # `HeNormal` is now concrete and fully instantiated
    def calc_fraction(self):
        return _NormalInitializer.calc_fraction(self)

    def initialize_initializer(self):
        return _NormalInitializer.initialize_initializer(self)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _HeBaseInitializer.set_shape(self, shape=shape)


class HeUniform(_HeBaseInitializer, _UniformInitializer):
    def __init__(self,
                 shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _HeBaseInitializer.__init__(self, shape=shape)
        _UniformInitializer.__init__(self)

    # `HeUniform` is now concrete and fully instantiated
    def calc_fraction(self):
        return _UniformInitializer.calc_fraction(self)

    def initialize_initializer(self):
        return _UniformInitializer.initialize_initializer(self)

    def set_shape(self, shape: Union[int, Tuple[int, ...]] = CONSTANT_DEFAULT_SHAPE[0]):
        _HeBaseInitializer.set_shape(self, shape=shape)
