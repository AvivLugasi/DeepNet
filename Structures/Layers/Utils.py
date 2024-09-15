from typing import Union
import numpy as np
import cupy as cp


def im2col(x: Union[np.ndarray, cp.ndarray],
           filter_dim: int = 3,
           stride: int = 1):
    depth = 1 if len(x.shape) == 2 else x.shape[2]

    rows = x.shape[0]
    cols = x.shape[1]

    output_shape = (rows - filter_dim) // stride + 1
    rows_iter = np.arange(0, rows - filter_dim + 1, stride)
    cols_iter = np.arange(0, cols - filter_dim + 1, stride)
    output = np.zeros((filter_dim * filter_dim * depth, output_shape * output_shape))
    index = 0

    for row in rows_iter:
        for col in cols_iter:
            window = x[row:row + filter_dim, col:col + filter_dim, :depth].reshape(-1)
            output[:, index] = window
            index += 1
    return output


def col2im(x: Union[np.ndarray, cp.ndarray],
           filter_dim: int = 3,
           stride: int = 1):
    pass
