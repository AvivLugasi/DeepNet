from typing import Union, Iterable
import cupy as cp
import numpy as np
import Structures.BaseStructures.Tensor as Tensor
from System.Errors.TensorErrors import TensorMatMulException


def _init_array(data: Union[np.ndarray, cp.ndarray, Iterable],
                is_gpu_enabled: bool = False,
                copy: bool = True,
                dtype: Union[np.dtype, cp.dtype] = None):
    if is_gpu_enabled:
        x = cp.array(obj=data,
                     copy=copy,
                     dtype=dtype)
    else:
        x = np.array(object=data,
                     copy=copy,
                     dtype=dtype)
    return x


def _matmul_validation(t1: Tensor, t2: Tensor):
    try:
        if cp.equal(t1.get_shape()[1], t2.get_shape()[0]):
            return True
        return "cols dimension of first matrix aren't the same as the rows dimension of the second matrix"
    except Exception as e:
        return f"An error occurred during matrices product validation: {e}"


class Tensor:

    def __init__(self,
                 data: Union[np.ndarray, cp.ndarray, Iterable],
                 is_gpu_enabled: bool = False,
                 copy: bool = True,
                 dtype: Union[np.dtype, cp.dtype] = None
                 ):
        self.__data = _init_array(data=data,
                                  is_gpu_enabled=is_gpu_enabled,
                                  copy=copy,
                                  dtype=dtype)
        self.is_on_gpu = is_gpu_enabled

    def __getitem__(self, index):
        return self.__data[index]

    def __setitem__(self, key, value):
        self.__data.__setitem__(slices=key,
                                y=value)

    def __repr__(self):
        return f"Tensor(data={self.__data}, shape={self.__data.shape}, type=d{self.__data.dtype})"

    def set_data(self,
                 x: Union[np.array, Iterable]):
        self.__data = x

    def get_data(self):
        return self.__data

    def get_shape(self):
        return self.__data.shape

    def get_size(self):
        return self.__data.size

    def mat_product(self,
                    other: Tensor,
                    perform_action_on_gpu: bool = True):
        validation_flag = _matmul_validation(self, other)

        if validation_flag == True:
            try:
                if perform_action_on_gpu:
                    return cp.asarray(self.__data) @ cp.asarray(other.get_data())
                else:
                    return cp.asnumpy(self.__data) @ cp.asnumpy(other.get_data())
            except Exception as e:
                raise TensorMatMulException(e)
        raise TensorMatMulException(validation_flag)
