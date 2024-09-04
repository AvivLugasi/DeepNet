from abc import ABC, abstractmethod
from typing import Union
import cupy as cp
import numpy as np


class Layer(ABC):
    def forward_pass(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def backward_pass(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args, **kwargs):
        return self.forward_pass(*args, **kwargs)

    def update_weights(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")