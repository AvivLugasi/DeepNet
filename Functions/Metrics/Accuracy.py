from typing import Union
from Functions.Metrics.Metric import Metric
import cupy as cp
import numpy as np


class Accuracy(Metric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def score(self,
              ground_truth: Union[np.ndarray, cp.ndarray],
              predictions: Union[np.ndarray, cp.ndarray]):
        xp = cp.get_array_module(predictions)

        # there are more than 2 classes
        if predictions.shape[0] > 1:
            predicted_labels = cp.argmax(predictions,
                                         axis=0)
            correct_labels = xp.argmax(ground_truth, axis=0)
        else:
            predicted_labels = xp.where(predictions > self.threshold, 1, 0)
            correct_labels = xp.where(ground_truth > self.threshold, 1, 0)

        correctly_classified = xp.sum(predicted_labels == correct_labels)
        return correctly_classified / predictions.shape[1]

    def config(self):
        return self.__class__.__name__
