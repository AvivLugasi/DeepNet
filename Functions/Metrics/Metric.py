from abc import ABC
from typing import Union
import numpy as np
import cupy as cp


class Metric(ABC):
    def score(self,
              ground_truth: Union[np.ndarray, cp.ndarray],
              predictions: Union[np.ndarray, cp.ndarray]):
        raise NotImplementedError("Subclasses should implement this method.")
