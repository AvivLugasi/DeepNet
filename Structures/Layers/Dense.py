from Functions.Activations.Linear import Linear
from Functions.Activations.Activation import Activation
from Functions.Activations.Consts import ACTIVATION_FUNCTION_VALID_VALUES
from Functions.Activations.Utils import return_activation_from_str
from Initializers.Consts import INITIALIZERS_VALID_VALUES
from Initializers.Initializer import Initializer
from Initializers.GlorotHeInitializers import GlorotUniform
from Initializers.Utils import return_initializer_from_str
from Initializers.Zeroes import Zeroes
import cupy as cp
import numpy as np
from typing import Union, Literal

from Optimizers.Optimizer import Optimizer
from Regularization.Regularizer import Regularizer
from Structures.Layers.BatchNorm import BatchNorm
from Structures.Layers.Layer import Layer
from System.Utils.Validations import validate_xp_module, validate_positive_int, validate_bool_val
from Structures.Layers.Consts import DEFAULT_DENSE_LAYER_UNITS


class Dense(Layer):
    def __init__(self,
                 units: int = DEFAULT_DENSE_LAYER_UNITS,
                 activation: Union[Literal[ACTIVATION_FUNCTION_VALID_VALUES], Activation] = Linear(),
                 use_bias: bool = True,
                 weights_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = GlorotUniform(),
                 bias_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Zeroes(),
                 weights_regularizer: Regularizer = None,
                 xp_module=cp,
                 batchnorm: BatchNorm = None):
        self.xp_module = validate_xp_module(xp=xp_module)

        self.units = validate_positive_int(units)

        self.activation_func = activation if isinstance(activation, Activation) \
            else return_activation_from_str(func_name=activation)

        self.weights_mat_init_method = weights_init_method if isinstance(weights_init_method, Initializer) \
            else return_initializer_from_str(initializer_name=weights_init_method)

        self.weights_mat = None
        # store the velocity value for the bias vector, incase momentum is applied
        self.v_weights = 0

        if use_bias:
            self.bias_mat = self._init_bias_mat(bias_init_method=bias_init_method)
            # store the velocity value for the bias vector, incase momentum is applied
            self.v_bias = 0
        else:
            self.bias_mat = None

        self.weights_regularizer = weights_regularizer

        self.input_mat = None
        self.weighted_input_mat = None
        self.activated_weighted_input = None
        self.gradient = None
        self.batchnorm = batchnorm

    def _init_bias_mat(self, bias_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer]):
        bias_mat_init_method = bias_init_method if isinstance(bias_init_method, Initializer) \
            else return_initializer_from_str(initializer_name=bias_init_method)
        bias_mat_init_method.set_shape(shape=(self.units, 1))
        return bias_mat_init_method.initialize(xp=self.xp_module)

    def init_weights_mat(self, previous_layer_units: int = 1):
        shape = (self.units, validate_positive_int(units=previous_layer_units))
        self.weights_mat_init_method.set_shape(shape=shape)
        self.weights_mat = self.weights_mat_init_method.initialize(xp=self.xp_module)

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        self.input_mat = input_mat
        if self.weights_mat is None:
            self.init_weights_mat(input_mat.shape[0])
        if self.bias_mat is not None:
            bias_term = self.bias_mat
        else:
            bias_term = 0
        self.weighted_input_mat = self.xp_module.dot(self.weights_mat, input_mat) + bias_term
        if self.batchnorm is not None:
            self.weighted_input_mat = self.batchnorm.forward_pass(input_mat=self.weighted_input_mat)
        self.activated_weighted_input = self.activation_func.activate(self.weighted_input_mat)
        if self.weights_regularizer is None:
            regularizer_cost = 0
        else:
            regularizer_cost = self.weights_regularizer.cost(self.weights_mat)
        return [self.activated_weighted_input, regularizer_cost]

    def backward_pass(self, grads: Union[np.ndarray, cp.ndarray], optimizer):
        hidden_layer_error = grads * self.activation_func.derivative(x=self.weighted_input_mat,
                                                                     optimizer=optimizer,
                                                                     grads=grads)
        if self.batchnorm is not None:
            hidden_layer_error = self.batchnorm.backward_pass(grads=hidden_layer_error, optimizer=optimizer)
        weights_gradients = self.xp_module.dot(hidden_layer_error, self.input_mat.T) / self.input_mat.shape[1]
        gradient_to_return = self.xp_module.dot(self.weights_mat.T, hidden_layer_error)
        self.update_weights(bias_gradients=cp.mean(hidden_layer_error, axis=1, keepdims=True),
                            weights_gradients=weights_gradients,
                            optimizer=optimizer)
        return gradient_to_return

    def update_weights(self, **kwargs):
        optimizer = kwargs.get('optimizer')
        weights_gradients = kwargs.get('weights_gradients')
        if self.bias_mat is not None:
            bias_gradients = kwargs.get('bias_gradients')
        if self.weights_regularizer is None:
            regularizer_term = 0
        else:
            regularizer_term = self.weights_regularizer(self.weights_mat)
        if optimizer is not None and isinstance(optimizer, Optimizer) and weights_gradients is not None:
            self.weights_mat, self.v_weights = optimizer.apply_gradients(gradients=weights_gradients,
                                                                         variables=self.weights_mat,
                                                                         regularizer=regularizer_term,
                                                                         velocity=self.v_weights)
            if self.bias_mat is not None:
                if bias_gradients is not None:
                    self.bias_mat, self.v_bias = optimizer.apply_gradients(gradients=bias_gradients,
                                                                           variables=self.bias_mat,
                                                                           velocity=self.v_bias)
                else:
                    raise TypeError("Missing required keyword argument: 'bias_gradients'")
        else:
            raise TypeError("Missing required keyword arguments: 'optimizer' and or 'weights_gradients'")

    def set_mod(self, is_training: bool):
        if validate_bool_val(val=is_training):
            if self.batchnorm is not None:
                self.batchnorm.set_mod(is_training=is_training)
        else:
            raise ValueError(f"is_training must be a bool, instead got {is_training}")
