from Functions.Losses.Loss import Loss
from typing import Union
import numpy as np
import cupy as cp
from System.Utils.Validations import validate_same_device_for_data_items


class MSE(Loss):
    def loss(self,
             ground_truth: Union[np.ndarray, cp.ndarray],
             predictions: Union[np.ndarray, cp.ndarray]):
        if validate_same_device_for_data_items(ground_truth=ground_truth, predictions=predictions):
            xp = cp.get_array_module(ground_truth)
            return 0.5 * xp.mean(xp.power(ground_truth - predictions, 2))

    def loss_derivative(self,
                        ground_truth: Union[np.ndarray, cp.ndarray],
                        predictions: Union[np.ndarray, cp.ndarray]):
        if validate_same_device_for_data_items(ground_truth=ground_truth, predictions=predictions):
            return (predictions - ground_truth) / ground_truth.shape[-1]
