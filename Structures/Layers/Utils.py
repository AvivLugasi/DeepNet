from typing import Union, Literal, Tuple
import numpy as np
import cupy as cp
import math
from Structures.Layers.Consts import CONVOLUTION_PADDING_VALID_VALUES
from System.Utils.Validations import is_whole_num


def is_valid_filter_params(input_mat_size: int,
                           filter_dim: int,
                           strides: int,
                           padding: int = 0):
    activation_window_size = get_activation_window_size(input_mat_size, filter_dim, strides, padding)
    if is_whole_num(activation_window_size):
        return True
    return False


def get_activation_window_size(input_mat_size: int,
                               filter_dim: int,
                               strides: int,
                               padding: int = 0):
    return (input_mat_size + 2 * padding - filter_dim) / strides + 1


def padding_required_for_same(input_mat_size: int,
                              filter_dim: int,
                              strides: int):
    padding = 0.5 * (strides * (input_mat_size - 1) - input_mat_size + filter_dim)
    if not is_whole_num(padding):
        padding = math.ceil(padding)
    return padding


def get_padding_from_str(padding_mode: Literal[CONVOLUTION_PADDING_VALID_VALUES],
                         input_mat_size: int,
                         filter_dim: int,
                         strides: int):
    if padding_mode == CONVOLUTION_PADDING_VALID_VALUES[0]:
        return 0
    elif padding_mode == CONVOLUTION_PADDING_VALID_VALUES[1]:
        return padding_required_for_same(input_mat_size, filter_dim, strides)
    else:
        raise ValueError(f"padding_mode must be one of {CONVOLUTION_PADDING_VALID_VALUES}"
                         f"instead got, {padding_mode}")


def get_input_dim(x: Union[np.ndarray, cp.ndarray],
                  input_depth_index: int,
                  batch_size_index: int):
    input_dim_no_depth = x.shape[:input_depth_index] + x.shape[input_depth_index + 1:]
    input_size = input_dim_no_depth[1]
    batch_size = x.shape[batch_size_index]
    depth = x.shape[input_depth_index]
    return batch_size, input_size, depth


def im2col(x: Union[np.ndarray, cp.ndarray],
           filter_dim: int = 3,
           strides: int = 1,
           depth_index: int = 3):
    batch_size, input_size, depth = get_input_dim(x, depth_index, 0)
    xp_module = cp.get_array_module(x)
    output_shape = (input_size - filter_dim) // strides + 1
    mat_iter = xp_module.arange(0, input_size - filter_dim + 1, strides)
    output = xp_module.zeros((batch_size, filter_dim * filter_dim * depth, output_shape * output_shape))
    window_index = 0
    sample_index = 0
    for sample in range(batch_size):
        for row in mat_iter:
            for col in mat_iter:
                window = x[sample_index, row:row + filter_dim, col:col + filter_dim, :depth].reshape(-1)
                output[sample_index, :, window_index] = window
                window_index += 1
        window_index = 0
        sample_index += 1
    return output


def col2im(col_mat: Union[np.ndarray, cp.ndarray],
           input_shape: Tuple,
           filter_dim: int = 3,
           stride: int = 1,
           padding: int = 0):
    xp_module = cp.get_array_module(col_mat)
    batch_size, height, width, depth = input_shape
    output = xp_module.zeros(input_shape)
    for batch in range(batch_size):
        for window_index in range(len(col_mat.shape[2])):
            window = col_mat[batch, :, window_index]
            # for d in range(depth):
            #     output[batch, , , d]



# x = np.array([[[1, 2, 3, 4, 5],
#                [6, 7, 8, 9, 10],
#                [11, 12, 13, 14, 15],
#                [16, 17, 18, 19, 20],
#                [21, 22, 23, 24, 25]]])
# x = x.reshape(1, 5, 5, 1)
# print(x.shape)
# out = im2col(x, 3, 2, 3)
# print(out.shape)
# print(out)
