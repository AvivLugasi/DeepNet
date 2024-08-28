from typing import Union
import cupy as cp
import numpy as np

EPSILON = 1e-15


def mean_subtraction(input_mat: Union[np.ndarray, cp.ndarray],
                     axis: int = 1):
    xp = cp.get_array_module(input_mat)
    input_mat -= xp.mean(input_mat, axis=axis, keepdims=True)
    return input_mat


def std_div_normalization(input_mat: Union[np.ndarray, cp.ndarray],
                          axis: int = 1):
    xp = cp.get_array_module(input_mat)
    std = xp.std(input_mat, axis=axis, keepdims=True)
    # avoid division by 0 if std is 0
    std = xp.where(std == 0, EPSILON, std)
    input_mat /= std
    return input_mat


def minmax_scalar(input_mat: Union[np.ndarray, cp.ndarray],
                  axis: int = 1):
    xp = cp.get_array_module(input_mat)
    min_val = xp.min(input_mat, axis=axis, keepdims=True)
    max_val = xp.max(input_mat, axis=axis, keepdims=True)
    # avoid division by 0 if max = min
    ranged_val = xp.where(max_val - min_val == 0, 1, max_val - min_val)
    input_mat = (input_mat - min_val) / ranged_val
    return input_mat


def standardization(input_mat: Union[np.ndarray, cp.ndarray],
                    axis: int = 1):
    zero_centered_input = mean_subtraction(input_mat, axis)
    scaled_by_std_input = std_div_normalization(zero_centered_input, axis)
    return scaled_by_std_input
