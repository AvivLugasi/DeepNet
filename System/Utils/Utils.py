from typing import Union
import numpy as np
import cupy as cp
from Functions.Activations.Sigmoid import Sigmoid
from Structures.Layers.SoftMax import softmax

EPSILON = 1e-15


def clip_predictions_in_given_range(predictions: Union[np.ndarray, cp.ndarray]):
    # Clip to prevent log(0) which would cause issues
    xp = cp.get_array_module(predictions)
    return xp.clip(predictions, EPSILON, 1 - EPSILON)


def logits_to_probabilities(predictions: Union[np.ndarray, cp.ndarray]):
    # Clip to prevent log(0) which would cause issues
    if predictions.shape[0] == 1:
        return Sigmoid().activate(predictions)
    else:
        return softmax(predictions)
