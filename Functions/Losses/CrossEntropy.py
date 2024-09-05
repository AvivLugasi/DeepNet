from Functions.Losses.Loss import Loss
from typing import Union
import numpy as np
import cupy as cp

from Structures.Layers.Consts import CROSS_ENTROPY_AFTER_SOFTMAX_FUNC
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
                        predictions: Union[np.ndarray, cp.ndarray],
                        softmax_indicator: str = None):
        if validate_same_device_for_data_items(ground_truth=ground_truth, predictions=predictions):
            if self.come_from_logits:
                predictions = logits_to_probabilities(predictions)
            predictions = clip_predictions_in_given_range(predictions)
            # incase this loss function comes after a softmax layer the loss'(P)*softmax'(Y) can be
            # written as predictions - ground_truth
            # P = output of softmax layer
            # Y = input of softmax layer
            # if this is not the case the standard derivative of loss'(P) is returned and the jacobian of
            # the softmax layer will be calculated
            if softmax_indicator == CROSS_ENTROPY_AFTER_SOFTMAX_FUNC:
                return predictions - ground_truth
            else:
                return -1 / predictions

    def config(self):
        return self.__class__.__name__
