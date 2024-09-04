from typing import Union
import numpy as np
import cupy as cp


def feature_one_hot(mat: Union[np.ndarray, cp.ndarray],
                    feature_col: int = None,
                    feature_row: int = None):
    xp = cp.get_array_module(mat)
    if feature_col is None and feature_row is not None:
        num_of_samples = mat.shape[-1]
        one_hot = xp.zeros(shape=[len(xp.unique(mat[feature_row, :])), num_of_samples], dtype=int)
        one_hot[mat[feature_row, :], xp.arange(0, num_of_samples)] = 1
    elif feature_col is not None and feature_row is None:
        num_of_samples = mat.shape[0]
        one_hot = xp.zeros(shape=[num_of_samples, len(xp.unique(mat[:, feature_col]))], dtype=int)
        one_hot[xp.arange(0, num_of_samples), mat[:, feature_col]] = 1
    else:
        raise ValueError("one of feature_col and feature_row must be defined the another one must remain None,"
                         f"instead got feature_col {feature_col}, feature_row {feature_row}")

    return one_hot
