from abc import abstractmethod, ABC
from Structures.Layers.Layer import Layer
from typing import Union
import cupy as cp
import numpy as np
from System.Utils.Validations import validate_number_in_range, validate_bool_val


def _set_keep_prob(keep_prob: float):
    if validate_number_in_range(n=keep_prob):
        return keep_prob
    raise ValueError(f"keep_prob must be in range of 0 to 1, instead got {keep_prob}")


class _DropoutBase(Layer, ABC):
    def __init__(self,
                 keep_prob: float = 0.5,
                 seed=None):
        self.set_keep_prob(keep_prob)
        self.seed = seed
        self._training_mod = True

    def backward_pass(self, *args, **kwargs):
        return kwargs.get('grads')

    def binary_mask(self, input_mat: Union[np.ndarray, cp.ndarray]):
        xp = cp.get_array_module(input_mat)
        xp.random.seed(seed=self.seed)
        dropout_mat = xp.random.binomial(n=1,
                                         p=self._keep_prob,
                                         size=input_mat.shape)
        input_mat_masked = xp.multiply(input_mat, dropout_mat)
        return input_mat_masked

    def set_keep_prob(self, keep_prob: float = 0.5):
        self._keep_prob = _set_keep_prob(keep_prob)

    def get_keep_prob(self):
        return self._keep_prob

    def get_mod(self):
        return self._training_mod

    def set_mod(self, is_training: bool):
        if validate_bool_val(val=is_training):
            self._training_mod = is_training
        else:
            raise ValueError(f"is_training must be a bool, instead got {is_training}")


class Dropout(_DropoutBase):
    def __init__(self, keep_prob: float = 0.5, seed=None):
        super().__init__(keep_prob, seed)

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        if self.get_mod():
            input_mat = self.binary_mask(input_mat=input_mat)
        else:
            input_mat *= self.get_keep_prob()
        return input_mat


class InvertedDropout(_DropoutBase):
    def __init__(self, keep_prob: float = 0.5, seed=None):
        super().__init__(keep_prob, seed)

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        if self.get_mod():
            input_mat = self.binary_mask(input_mat=input_mat)
            input_mat /= self.get_keep_prob()
        return input_mat
