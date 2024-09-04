from Functions.Losses.Loss import Loss
from typing import Union
import numpy as np
import cupy as cp
from System.Utils.Utils import logits_to_probabilities, clip_predictions_in_given_range
from System.Utils.Validations import validate_same_device_for_data_items


class CrossEntropy(Loss):
    def __init__(self, come_from_logits: bool = False):
        self.come_from_logits = come_from_logits

    def loss(self,
             ground_truth: Union[np.ndarray, cp.ndarray],
             predictions: Union[np.ndarray, cp.ndarray]):
        if validate_same_device_for_data_items(ground_truth=ground_truth, predictions=predictions):
            xp = cp.get_array_module(predictions)
            if self.come_from_logits:
                predictions = logits_to_probabilities(predictions)
            predictions = clip_predictions_in_given_range(predictions)
            return xp.mean(-xp.sum(ground_truth * xp.log(predictions), axis=0))

    def loss_derivative(self,
                        ground_truth: Union[np.ndarray, cp.ndarray],
                        predictions: Union[np.ndarray, cp.ndarray]):
        if validate_same_device_for_data_items(ground_truth=ground_truth, predictions=predictions):
            if self.come_from_logits:
                predictions = logits_to_probabilities(predictions)
            predictions = clip_predictions_in_given_range(predictions)
            # not taking into account usage with softmax
            #return -1 / predictions
            return predictions - ground_truth

    def config(self):
        return self.__class__.__name__
