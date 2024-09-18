from typing import Union, Literal, Tuple
from Functions.Activations.Activation import Activation
from Functions.Activations.Consts import ACTIVATION_FUNCTION_VALID_VALUES
from Functions.Activations.LeakyRelu import LeakyRelu
from Functions.Activations.Relu import Relu
from Functions.Activations.Utils import return_activation_from_str
from Initializers.Consts import INITIALIZERS_VALID_VALUES
from Initializers.GlorotHeInitializers import GlorotUniform
from Initializers.Initializer import Initializer
from Initializers.Utils import return_initializer_from_str
from Initializers.Zeroes import Zeroes
from Optimizers.SGD import SGD
from Optimizers.Schedules.ExponentialDecay import ExponentialDecay
from Regularization.Regularizer import Regularizer
from Structures.Layers.Consts import CONVOLUTION_PADDING_VALID_VALUES
from Structures.Layers.Layer import Layer
import cupy as cp
import numpy as np
from Structures.Layers.Utils import get_padding_from_str, im2col, get_input_dim, get_activation_window_size, col2im
from System.Utils.Validations import validate_xp_module, validate_positive_int, is_whole_num


class Conv(Layer):
    def __init__(self,
                 num_of_filters: int = 1,
                 filter_size: int = 3,
                 strides: int = 1,
                 is_channels_last: bool = True,
                 padding: Union[Literal[CONVOLUTION_PADDING_VALID_VALUES], int] = 0,
                 activation: Union[Literal[ACTIVATION_FUNCTION_VALID_VALUES], Activation] = Relu(),
                 use_bias: bool = True,
                 filters_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = GlorotUniform(),
                 bias_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer] = Zeroes(),
                 filters_regularizer: Regularizer = None,
                 samples_dim_index: int = 0,
                 xp_module=cp,
                 ):
        self.xp_module = validate_xp_module(xp=xp_module)
        self.samples_dim_index = samples_dim_index
        self.num_of_filters = validate_positive_int(num_of_filters)
        self.input_depth_index = 3 if is_channels_last else 1
        self.filter_size = validate_positive_int(filter_size)
        self.padding = padding
        self.strides = validate_positive_int(strides)
        self.activation_func = activation if isinstance(activation, Activation) \
            else return_activation_from_str(func_name=activation)

        self.filters_init_method = filters_init_method if isinstance(filters_init_method, Initializer) \
            else return_initializer_from_str(initializer_name=filters_init_method)

        self.filters_mat = None
        # store the velocity value for the filter vector, incase momentum is applied
        self.v_filters = 0

        if use_bias:
            self.bias_mat = self._init_bias_mat(bias_init_method=bias_init_method)
            # store the velocity value for the bias vector, incase momentum is applied
            self.v_bias = 0
        else:
            self.bias_mat = None

        self.filters_regularizer = filters_regularizer

        self.input_mat_im2col = None
        self.activation_map_image_shape = None
        self.batch_size = None
        self.num_of_windows_per_sample = None
        self.input_dim = None

    def _init_bias_mat(self, bias_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer]):
        bias_mat_init_method = bias_init_method if isinstance(bias_init_method, Initializer) \
            else return_initializer_from_str(initializer_name=bias_init_method)
        bias_mat_init_method.set_shape(shape=(1, self.num_of_filters))
        return bias_mat_init_method.initialize(xp=self.xp_module)

    def _init_filters_mat(self, previous_layer_shape: Tuple):
        self.input_depth = previous_layer_shape[self.input_depth_index]
        shape = (self.filter_size * self.filter_size * self.input_depth, self.num_of_filters)
        self.filters_init_method.set_shape(shape=shape)
        self.filters_mat = self.filters_init_method.initialize(xp=self.xp_module)

    def _add_padding(self, input_mat: Union[np.ndarray, cp.ndarray]):

        input_mat = self.xp_module.pad(input_mat,
                                       ((0, 0),
                                        (self.padding, self.padding),
                                        (self.padding, self.padding),
                                        (0, 0)),
                                       mode='constant')
        return input_mat

    def _get_padding_required(self, input_mat_size):
        if isinstance(self.padding, str):
            self.padding = int(get_padding_from_str(padding_mode=self.padding,
                                                    input_mat_size=input_mat_size,
                                                    filter_dim=self.filter_size,
                                                    strides=self.strides))
        return self.padding

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        if self.filters_mat is None:
            self._init_filters_mat(input_mat.shape)
        if self.bias_mat is not None:
            bias_term = self.bias_mat
        else:
            bias_term = 0
        self.input_dim = input_mat.shape
        self.padding = self._get_padding_required(input_mat.shape[2])
        output_size = get_activation_window_size(input_mat_size=input_mat.shape[2],
                                                 filter_dim=self.filter_size,
                                                 strides=self.strides,
                                                 padding=self.padding)
        if not is_whole_num(output_size):
            raise ValueError(f"The given paramters for strides, padding and filter size"
                             f"{self.strides}, {self.padding}, {self.filter_size} are not valid because"
                             f"the output shape is not whole {output_size}")
        output_size = int(output_size)
        if self.padding != 0:
            input_mat = self._add_padding(input_mat)
        depth = input_mat.shape[self.input_depth_index]
        self.batch_size = input_mat.shape[self.samples_dim_index]
        self.input_mat_im2col = im2col(x=input_mat,
                                       filter_dim=self.filter_size,
                                       strides=self.strides)
        activation_map = self.xp_module.dot(self.xp_module.transpose(self.input_mat_im2col, (0, 2, 1)),
                                            self.filters_mat) + bias_term
        self.num_of_windows_per_sample = activation_map.shape[1]
        new_shape = (self.batch_size, output_size, output_size, depth) if self.input_depth_index == 3 \
            else (self.batch_size, depth, output_size, output_size)
        self.activation_map_image_shape = activation_map.reshape(new_shape)
        output = self.activation_func.activate(self.activation_map_image_shape)
        if self.filters_regularizer is None:
            regularizer_cost = 0
        else:
            regularizer_cost = self.filters_regularizer.cost(self.filters_mat)
        return [output, regularizer_cost]

    def backward_pass(self, grads: Union[np.ndarray, cp.ndarray], optimizer):
        d_f = grads * self.activation_func.derivative(x=self.activation_map_image_shape,
                                                      optimizer=optimizer,
                                                      grads=grads)
        d_f = d_f.reshape(self.batch_size, self.num_of_windows_per_sample, self.num_of_filters)
        d_b = self.xp_module.mean(d_f, axis=(0, 1)).reshape(1, 3)
        d_w = self.xp_module.einsum('ijk,ikl->jl', self.input_mat_im2col, d_f) / self.batch_size
        d_x = self.xp_module.transpose(self.xp_module.dot(d_f, self.filters_mat.T), (0, 2, 1))
        self.update_weights(bias_gradients=d_b,
                            weights_gradients=d_w,
                            optimizer=optimizer)
        d_x = col2im(col_mat=d_x,
                     input_shape=self.input_dim,
                     filter_dim=self.filter_size,
                     stride=self.strides,
                     padding=self.padding)
        return d_x

    def update_weights(self, **kwargs):
        optimizer = kwargs.get('optimizer')
        weights_gradients = kwargs.get('weights_gradients')
        if self.bias_mat is not None:
            bias_gradients = kwargs.get('bias_gradients')
        if self.filters_regularizer is None:
            regularizer_term = 0
        else:
            regularizer_term = self.filters_regularizer(self.filters_mat)
        self.filters_mat, self.v_filters = optimizer.apply_gradients(gradients=weights_gradients,
                                                                     variables=self.filters_mat,
                                                                     velocity=self.v_filters,
                                                                     regularizer=regularizer_term, )
        if self.bias_mat is not None:
            if bias_gradients is not None:
                self.bias_mat, self.v_bias = optimizer.apply_gradients(gradients=bias_gradients,
                                                                       variables=self.bias_mat,
                                                                       velocity=self.v_bias)
            else:
                raise TypeError("Missing required keyword argument: 'bias_gradients'")

    def set_mod(self, *args, **kwargs):
        pass
