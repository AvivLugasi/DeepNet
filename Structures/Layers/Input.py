from typing import Tuple, Union
import numpy as np
import cupy as cp
from Structures.Layers.Layer import Layer
from System.Utils.Validations import validate_positive_int, validate_np_cp_array, validate_same_device_for_data_items
from collections import deque


class Input(Layer):
    def __init__(self,
                 shape: Tuple[int, ...] = None,
                 data_x: Union[np.ndarray, cp.ndarray] = None,
                 data_y: Union[np.ndarray, cp.ndarray] = None,
                 batch_size: int = 1,
                 shuffle_in_every_epoch: bool = True,
                 features_are_rows: bool = False):
        if data_x is not None:
            validate_np_cp_array(data_x)
        if data_y is not None:
            validate_np_cp_array(data_y)
        self.shape = shape
        self.data_x = data_x
        self.data_y = data_y
        if validate_positive_int(batch_size):
            self.batch_size = batch_size
        self.shuffle_in_every_epoch = shuffle_in_every_epoch
        self._features_are_rows = features_are_rows
        self.batches_queue = None

    def set_data(self, data_x: Union[np.ndarray, cp.ndarray], data_y: Union[np.ndarray, cp.ndarray]):
        if data_x is not None and data_y is not None:
            validate_same_device_for_data_items(data_x = data_x, data_y = data_y)
            validate_np_cp_array(data_x)
            xp = validate_np_cp_array(data_y)
            if self._features_are_rows:
                self.data_x = data_x
                self.data_y = data_y
            else:
                self.data_x = xp.transpose(data_x)
                self.data_y = xp.transpose(data_y)
            self.shape = self.data_x.shape
        else:
            raise ValueError("Data cant be None, a np/cp nd.array should be passed")

    def _shuffle(self):
        permutation = np.random.permutation(self.data_x.shape[-1])
        self.data_x = self.data_x[:, permutation]
        self.data_y = self.data_y[:, permutation]

    def init_queue(self, input_mat_x: Union[np.ndarray, cp.ndarray] = None, input_mat_y: Union[np.ndarray, cp.ndarray] = None):
        if input_mat_x is not None and input_mat_y is not None:
            self.set_data(input_mat_x, input_mat_y)
        if self.shuffle_in_every_epoch:
            self.batches_queue = deque()
            self._shuffle()
        self.batches_queue = deque()
        for batch_x, batch_y in self._iter_batches():
            self.batches_queue.append([batch_x, batch_y])

    def _iter_batches(self):
        num_of_full_batches = self.data_x.shape[-1] // self.batch_size
        partial_batch_size = self.data_x.shape[-1] % self.batch_size

        # Iterate over full batches
        for i in range(num_of_full_batches):
            start_col = i * self.batch_size
            end_col = start_col + self.batch_size
            yield self.data_x[:, start_col:end_col], self.data_y[:, start_col:end_col]

        # Handle the last batch if there is a remainder
        if partial_batch_size > 0:
            yield self.data_x[:, -partial_batch_size:], self.data_y[:, -partial_batch_size:]

    def forward_pass(self, input_mat_x: Union[np.ndarray, cp.ndarray] = None, input_mat_y: Union[np.ndarray, cp.ndarray] = None):
        if not self.batches_queue:
            self.init_queue(input_mat_x, input_mat_y)
            if not self.batches_queue:
                raise RuntimeError("No batches in the queue")
        batch = self.batches_queue.popleft()
        self.batches_queue.append(batch)
        return batch

    def backward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        pass
