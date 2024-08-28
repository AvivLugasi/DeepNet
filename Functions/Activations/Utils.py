from typing import Literal
from Functions.Activations.Consts import ACTIVATION_FUNCTION_VALID_VALUES
from Functions.Activations.Activation import Activation
from Functions.Activations.Elu import Elu
from Functions.Activations.LeakyRelu import LeakyRelu
from Functions.Activations.Linear import Linear
from Functions.Activations.Relu import Relu
from Functions.Activations.Sigmoid import Sigmoid
from Functions.Activations.Softmax import Softmax
from Functions.Activations.Tanh import Tanh


def return_activation_from_str(func_name: Literal[ACTIVATION_FUNCTION_VALID_VALUES]) -> Activation:
    if func_name not in ACTIVATION_FUNCTION_VALID_VALUES:
        raise ValueError(f"func_name must be one of {ACTIVATION_FUNCTION_VALID_VALUES}, got {func_name} instead")
    else:
        if func_name == ACTIVATION_FUNCTION_VALID_VALUES[0]:
            return Relu()
        elif func_name == ACTIVATION_FUNCTION_VALID_VALUES[1]:
            return Elu()
        elif func_name == ACTIVATION_FUNCTION_VALID_VALUES[2]:
            return LeakyRelu()
        elif func_name == ACTIVATION_FUNCTION_VALID_VALUES[3]:
            return Sigmoid()
        elif func_name == ACTIVATION_FUNCTION_VALID_VALUES[4]:
            return Tanh()
        elif func_name == ACTIVATION_FUNCTION_VALID_VALUES[5]:
            return Softmax()
        return Linear()
