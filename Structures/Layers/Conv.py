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
from Regularization.Regularizer import Regularizer
from Structures.Layers.Consts import CONVOLUTION_PADDING_VALID_VALUES
from Structures.Layers.Layer import Layer
import cupy as cp
import numpy as np
from Structures.Layers.Utils import get_padding_from_str, im2col, get_input_dim, get_activation_window_size
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

        self.input_mat = None
        self.activation_map = None
        self.output = None

    def _init_bias_mat(self, bias_init_method: Union[Literal[INITIALIZERS_VALID_VALUES], Initializer]):
        bias_mat_init_method = bias_init_method if isinstance(bias_init_method, Initializer) \
            else return_initializer_from_str(initializer_name=bias_init_method)
        bias_mat_init_method.set_shape(shape=(1, self.num_of_filters))
        return bias_mat_init_method.initialize(xp=self.xp_module)

    def _init_filters_mat(self, previous_layer_shape: Tuple):
        shape = (self.filter_size * self.filter_size * previous_layer_shape[self.input_depth_index],
                 self.num_of_filters)
        self.filters_init_method.set_shape(shape=shape)
        self.filters_mat = self.filters_init_method.initialize(xp=self.xp_module)

    def _add_padding(self, input_mat: Union[np.ndarray, cp.ndarray]):
        batch_size, input_size, depth = get_input_dim(x=input_mat,
                                                      input_depth_index=self.input_depth_index,
                                                      batch_size_index=self.samples_dim_index)
        padding = self._get_padding_required(input_mat_size=input_size)
        shape = (batch_size, input_size, 1 * padding, depth) if self.input_depth_index == 3 \
            else (batch_size, depth, input_size, 1 * padding)
        add_right = self.xp_module.zeros(shape=shape, dtype=float)
        add_left = self.xp_module.zeros(shape=shape, dtype=float)
        input_mat = self.xp_module.concatenate((input_mat, add_right), axis=2)
        input_mat = self.xp_module.concatenate((add_left, input_mat), axis=2)
        input_dim_no_depth = input_mat.shape[:self.input_depth_index] + input_mat.shape[self.input_depth_index + 1:]
        input_size = input_dim_no_depth[2]
        shape = (batch_size, 1 * padding, input_size, depth) if self.input_depth_index == 3 \
            else (batch_size, depth, 1 * padding, input_size)
        add_up = self.xp_module.zeros(shape=shape, dtype=float)
        add_down = self.xp_module.zeros(shape=shape, dtype=float)
        input_mat = self.xp_module.concatenate((input_mat, add_up), axis=1)
        input_mat = self.xp_module.concatenate((add_down, input_mat), axis=1)
        return input_mat

    def _get_padding_required(self, input_mat_size):
        if isinstance(self.padding, str):
            self.padding = get_padding_from_str(padding_mode=self.padding,
                                                input_mat_size=input_mat_size,
                                                filter_dim=self.filter_size,
                                                strides=self.strides)
        return self.padding

    def forward_pass(self, input_mat: Union[np.ndarray, cp.ndarray]):
        self.input_mat = input_mat
        if self.filters_mat is None:
            self._init_filters_mat(input_mat.shape)
        if self.bias_mat is not None:
            bias_term = self.bias_mat
        else:
            bias_term = 0
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
        batch_size = input_mat.shape[self.samples_dim_index]
        input_mat_im2col = im2col(x=input_mat,
                                  filter_dim=self.filter_size,
                                  strides=self.strides)
        self.activation_map = self.xp_module.dot(input_mat_im2col, self.filters_mat) + bias_term
        new_shape = (batch_size, output_size, output_size, depth) if self.input_depth_index == 3 \
            else (batch_size, depth, output_size, output_size)
        activation_map_image_shape = self.activation_map.reshape(new_shape)
        self.output = self.activation_func.activate(activation_map_image_shape)
        if self.filters_regularizer is None:
            regularizer_cost = 0
        else:
            regularizer_cost = self.filters_regularizer.cost(self.filters_mat)
        return [self.output, regularizer_cost]

    def backward_pass(self, *args, **kwargs):
        pass

    def update_weights(self, **kwargs):
        pass

    def set_mod(self, *args, **kwargs):
        pass
