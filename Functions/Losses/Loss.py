from abc import ABC
from typing import Union
import numpy as np
import cupy as cp


class Loss(ABC):
    def loss(self,
             ground_truth: Union[np.ndarray, cp.ndarray],
             predictions: Union[np.ndarray, cp.ndarray]):
        raise NotImplementedError("Subclasses should implement this method.")

    def loss_derivative(self,
                        ground_truth: Union[np.ndarray, cp.ndarray],
                        predictions: Union[np.ndarray, cp.ndarray]):
        raise NotImplementedError("Subclasses should implement this method.")

    def config(self):
        raise NotImplementedError("Subclasses should implement this method.")
