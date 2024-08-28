from typing import Tuple, Union
import numpy as np
import cupy as cp
from Structures.Layers.Layer import Layer
from System.Utils.Validations import validate_positive_int, validate_np_cp_array
from collections import deque


class Input(Layer):
    def __init__(self,
                 shape: Tuple[int, ...] = None,
                 data: Union[np.ndarray, cp.ndarray] = None,
                 batch_size: int = 1,
                 shuffle_in_every_epoch: bool = True,
                 features_are_rows: bool = False):
        if data is not None:
            validate_np_cp_array(data)
        self.shape = shape
        self.data = data
        if validate_positive_int(batch_size):
            self.batch_size = batch_size
        self.shuffle_in_every_epoch = shuffle_in_every_epoch
        self._features_are_rows = features_are_rows
        self.batches_queue = deque()

    def set_data(self, data: Union[np.ndarray, cp.ndarray]):
        if data is not None:
            xp = validate_np_cp_array(data)
            if self._features_are_rows:
                self.data = data
            else:
                self.data = xp.transpose(data)
            self.shape = self.data.shape[0]
        else:
            raise ValueError("Data cant be None, a np/cp nd.array should be passed")

    def _shuffle(self):
        self.data = self.data[:, np.random.permutation(self.data.shape[-1])]

    def init_queue(self, input_mat: Union[np.ndarray, cp.ndarray] = None):
        if input_mat is not None:
            self.set_data(data=input_mat)
        if self.shuffle_in_every_epoch:
            self.batches_queue = deque()
            self._shuffle()

        if not self.batches_queue:
            for batch in self._iter_batches():
                self.batches_queue.append(batch)

    def _iter_batches(self):
        num_of_full_batches = self.data.shape[-1] // self.batch_size
        partial_batch_size = self.data.shape[-1] % self.batch_size

        # Iterate over full batches
        for i in range(num_of_full_batches):
            start_col = i * self.batch_size
            end_col = start_col + self.batch_size
            yield self.data[:, start_col:end_col]

        # Handle the last batch if there is a remainder
        if partial_batch_size > 0:
            yield self.data[:, -partial_batch_size:]

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray] = None):
        if not self.batches_queue:
            self.init_queue(input_mat)
            if not self.batches_queue:
                raise RuntimeError("No batches in the queue")
        batch = self.batches_queue.popleft()
        self.batches_queue.append(batch)
        return batch

    def backward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        pass
