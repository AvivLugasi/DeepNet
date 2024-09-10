from typing import Union, Literal
import numpy as np
import cupy as cp

from Initializers.Constants import Constants
from Initializers.Consts import INITIALIZERS_VALID_VALUES
from Initializers.Initializer import Initializer
from Initializers.Utils import return_initializer_from_str
from Initializers.Zeroes import Zeroes
from Optimizers.Optimizer import Optimizer
from PreProcessing.Normalization import EPSILON, standardization
from Structures.Layers.Consts import DEFAULT_DENSE_LAYER_UNITS
from Structures.Layers.Layer import Layer
from System.Utils.Validations import validate_positive_int, validate_xp_module, validate_bool_val, \
    validate_number_in_range


class BatchNorm(Layer):
    def __init__(self,
                 vectors_size: int = DEFAULT_DENSE_LAYER_UNITS,
                 axis: int = 1,
                 samples_size_index: int = 1,
                 beta_initializer: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Zeroes(),
                 gamma_initializer: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Constants(),
                 momentum: float = 0.9,
                 moving_mean_initializer: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Zeroes(),
                 moving_std_initializer: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Constants(),
                 xp_module=cp):
        # cache variables for back prop stage
        self.gemma_x = None
        self.batch_std = None
        self.batch_mean = None
        self.zero_centered_input = None
        self.input = None
        self.normalized_input = None
        self.scaled_Shifted_output = None
        # layer parameters
        self.xp_module = validate_xp_module(xp=xp_module)
        vectors_size = validate_positive_int(vectors_size)
        self.beta_vec = self._init_vector(initializer=beta_initializer, vectors_size=vectors_size)
        self.gamma_vec = self._init_vector(initializer=gamma_initializer, vectors_size=vectors_size)
        # velocity of gamma and beta
        self.v_gamma = 0
        self.v_beta = 0
        self.moving_mean = self._init_vector(initializer=moving_mean_initializer, vectors_size=vectors_size)
        self.moving_std = self._init_vector(initializer=moving_std_initializer, vectors_size=vectors_size)
        self.axis = axis
        self.samples_size_index = samples_size_index
        self._training_mod = True

        if validate_number_in_range(n=momentum,
                                    include_lower=True,
                                    include_upper=False):
            self.momentum = momentum
        else:
            raise ValueError(f"momentum must be in range [0, 1), instead got:{momentum}")

    def get_mod(self):
        return self._training_mod

    def set_mod(self, is_training: bool):
        if validate_bool_val(val=is_training):
            self._training_mod = is_training
        else:
            raise ValueError(f"is_training must be a bool, instead got {is_training}")

    def _init_vector(self,
                     initializer: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer],
                     vectors_size: int = DEFAULT_DENSE_LAYER_UNITS):
        initializer = initializer if isinstance(initializer, Initializer) \
            else return_initializer_from_str(initializer_name=initializer)
        initializer.set_shape(shape=(vectors_size, 1))
        return initializer.initialize(xp=self.xp_module)

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        self.input = input_mat
        if self._training_mod:
            # compute normalized input at train
            self.normalized_input, self.batch_mean, self.batch_std = standardization(input_mat=self.input,
                                                                                     axis=self.axis,
                                                                                     return_params=True)
            # update moving mean and std
            self.moving_mean = self.moving_mean * self.momentum + self.batch_mean * (1 - self.momentum)
            self.moving_std = self.moving_std * self.momentum + self.batch_std * (1 - self.momentum)
        else:
            # compute normalized input at test
            self.normalized_input = (self.input - self.moving_mean) / self.moving_std
        # scale and shift normalized input
        self.scaled_Shifted_output = self.gamma_vec * self.normalized_input + self.beta_vec
        return self.scaled_Shifted_output

    def backward_pass(self, grads: Union[np.ndarray, cp.ndarray], optimizer):
        # grads of learned parameters gamma and beta
        beta_grads = self.xp_module.sum(grads, axis=self.axis, keepdims=True)
        gamma_grads = self.xp_module.sum(self.normalized_input * grads, axis=self.axis, keepdims=True)
        # updating the parameters
        self.update_weights(gamma_grads=gamma_grads, beta_grads=beta_grads, optimizer=optimizer)
        # return gradient of x
        d_y_x_normalized = grads * self.gamma_vec
        inv_var = 1 / self.xp_module.sqrt(self.batch_std**2 + EPSILON)
        m = self.input.shape[self.samples_size_index]
        d_x = (1. / m) * inv_var * (m * d_y_x_normalized - self.xp_module.sum(d_y_x_normalized,
                                                                              axis=self.axis,
                                                                              keepdims=True)
                                    - self.normalized_input *
                                    self.xp_module.sum(d_y_x_normalized * self.normalized_input,
                                                       axis=self.axis,
                                                       keepdims=True))
        return d_x

    def update_weights(self,
                       gamma_grads: Union[np.ndarray, cp.ndarray],
                       beta_grads: Union[np.ndarray, cp.ndarray],
                       optimizer: Optimizer):
        self.gamma_vec, self.v_gamma = optimizer.apply_gradients(gradients=gamma_grads,
                                                                 variables=self.gamma_vec,
                                                                 velocity=self.v_gamma)
        self.beta_vec, self.v_beta = optimizer.apply_gradients(gradients=beta_grads,
                                                               variables=self.beta_vec,
                                                               velocity=self.v_beta)
