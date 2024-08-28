from typing import Literal
from Initializers.Consts import INITIALIZERS_VALID_VALUES
from Initializers.Initializer import Initializer
from Initializers.Constants import Constants
from Initializers.GlorotHeInitializers import GlorotUniform, GlorotNormal, HeNormal, HeUniform
from Initializers.Identity import Identity
from Initializers.RandomNormal import RandomNormal
from Initializers.RandomUniform import RandomUniform
from Initializers.Zeroes import Zeroes


def return_initializer_from_str(initializer_name: Literal[INITIALIZERS_VALID_VALUES]) -> Initializer:
    if initializer_name not in INITIALIZERS_VALID_VALUES:
        raise ValueError(f"initializer_name must be one of {INITIALIZERS_VALID_VALUES}, got {initializer_name} instead")
    else:
        if initializer_name == INITIALIZERS_VALID_VALUES[0]:
            return Constants()
        elif initializer_name == INITIALIZERS_VALID_VALUES[1]:
            return GlorotUniform()
        elif initializer_name == INITIALIZERS_VALID_VALUES[2]:
            return GlorotNormal()
        elif initializer_name == INITIALIZERS_VALID_VALUES[3]:
            return HeNormal()
        elif initializer_name == INITIALIZERS_VALID_VALUES[4]:
            return HeUniform()
        elif initializer_name == INITIALIZERS_VALID_VALUES[5]:
            return Identity()
        elif initializer_name == INITIALIZERS_VALID_VALUES[6]:
            return RandomNormal()
        elif initializer_name == INITIALIZERS_VALID_VALUES[7]:
            return RandomUniform()
        else:
            return Zeroes()
