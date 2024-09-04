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
from Structures.Layers.Layer import Layer
from System.Utils.Validations import validate_xp_module, validate_positive_int
from Structures.Layers.Consts import DEFAULT_DENSE_LAYER_UNITS


class Dense(Layer):
    def __init__(self,
                 units: int = DEFAULT_DENSE_LAYER_UNITS,
                 activation: Union[Literal[ACTIVATION_FUNCTION_VALID_VALUES], Activation] = Linear(),
                 use_bias: bool = True,
                 weights_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = GlorotUniform(),
                 bias_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Zeroes(),
                 weights_regularizer=None,
                 xp_module=cp):
        self.xp_module = validate_xp_module(xp=xp_module)

        self.units = validate_positive_int(units=units)

        self.activation_func = activation if isinstance(activation, Activation) \
            else return_activation_from_str(func_name=activation)

        self.weights_mat_init_method = weights_init_method if isinstance(weights_init_method, Initializer) \
            else return_initializer_from_str(initializer_name=weights_init_method)

        self.weights_mat = None

        if use_bias:
            self.bias_mat = self._init_bias_mat(bias_init_method=bias_init_method)
        else:
            self.bias_mat = None

        self.weights_regularizer = weights_regularizer

        self.input_mat = None
        self.weighted_input_mat = None
        self.activated_weighted_input = None
        self.gradient = None

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

        self.weighted_input_mat = self.xp_module.dot(self.weights_mat, input_mat) + self.bias_mat
        self.activated_weighted_input = self.activation_func.activate(self.weighted_input_mat)
        return self.activated_weighted_input

    def backward_pass(self, grads: Union[np.ndarray, cp.ndarray], optimizer):
        hidden_layer_error = grads * self.activation_func.derivative(x=self.weighted_input_mat)
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

        if optimizer is not None and isinstance(optimizer, Optimizer) and weights_gradients is not None:
            self.weights_mat = optimizer.apply_gradients(gradients=weights_gradients,
                                                         variables=self.weights_mat,
                                                         regularizer=self.weights_regularizer)
            if self.bias_mat is not None:
                if bias_gradients is not None:
                    self.bias_mat = optimizer.apply_gradients(gradients=bias_gradients,
                                                              variables=self.bias_mat,
                                                              regularizer=self.weights_regularizer)
                else:
                    raise TypeError("Missing required keyword argument: 'bias_gradients'")
        else:
            raise TypeError("Missing required keyword arguments: 'optimizer' and or 'weights_gradients'")
