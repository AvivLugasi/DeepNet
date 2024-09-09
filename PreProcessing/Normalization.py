from typing import Union
import cupy as cp
import numpy as np

EPSILON = 1e-15


def mean_subtraction(input_mat: Union[np.ndarray, cp.ndarray],
                     axis: int = 1,
                     return_computed_mean: bool = False):
    xp = cp.get_array_module(input_mat)
    mean = xp.mean(input_mat, axis=axis, keepdims=True)
    mean_subtracted = input_mat - mean
    if not return_computed_mean:
        return mean_subtracted
    return mean_subtracted, mean


def std_div_normalization(input_mat: Union[np.ndarray, cp.ndarray],
                          axis: int = 1,
                          return_computed_std: bool = False):
    xp = cp.get_array_module(input_mat)
    std = xp.std(input_mat, axis=axis, keepdims=True)
    # avoid division by 0 if std is 0
    std += EPSILON
    input_std_divided = input_mat / std
    if not return_computed_std:
        return input_std_divided
    return input_std_divided, std


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
                    axis: int = 1,
                    return_params: bool = False):
    _, mean = mean_subtraction(input_mat=input_mat,
                               axis=axis,
                               return_computed_mean=True)
    _, std = std_div_normalization(input_mat=input_mat,
                                   axis=axis,
                                   return_computed_std=True)
    standardized_input = (input_mat - mean) / (std + EPSILON)
    if return_params:
        return standardized_input, mean, std
    else:
        return standardized_input
