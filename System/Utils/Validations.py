import cupy as cp
import numpy as np


def validate_xp_module(xp):
    """
    Ensure xp is either numpy or cupy module
    :param xp: numpy or cupy module.
    :raise ValueError: if xp is none of cupy or numpy
    """
    if xp not in [np, cp]:
        raise ValueError("Parameter xp must be either numpy (np) or cupy (cp).")
    return xp


def validate_np_cp_array(data):
    if isinstance(data, np.ndarray):
        return np
    elif isinstance(data, cp.ndarray):
        return cp
    else:
        raise ValueError(f"data must be either a numpy ndarray or a cupy ndarray. Got: {type(data)}")


def validate_same_device_for_data_items(**data):
    """
    Validate that all data items are either numpy ndarrays or cupy ndarrays,
    and that all data items are on the same device (either numpy or cupy).

    :param data: Named data items to validate (expects ndarrays).
    :raise ValueError: if any item is not a numpy or cupy ndarray, or if items are on different devices.
    """
    # Initialize to None to track the type of the first array (either numpy or cupy)
    first_device = None

    for name, item in data.items():
        # Check if the item is an ndarray from numpy or cupy
        current_device = validate_np_cp_array(item)

        if first_device is None:
            first_device = current_device
        # Check that all items are on the same device (either numpy or cupy)
        if current_device != first_device:
            raise ValueError(f"Item '{name}' is not on the same device as the others. "
                             f"Expected all items to be {first_device}, but got {current_device}.")

    return True


def validate_positive_int(units: int) -> int:
    if units is None or units < 1 or not isinstance(units, int):
        raise ValueError(f"units must be a positive integer, got {units} instead")
    return units


def validate_number_in_range(n: float,
                             lower: float = 0,
                             upper: float = 1,
                             include_lower: bool = True,
                             include_upper: bool = True):
    if include_lower and not include_upper:
        return upper > n >= lower
    elif not include_lower and include_upper:
        return upper >= n > lower
    elif not include_lower and not include_upper:
        return upper > n > lower
    else:
        return upper >= n >= lower


def validate_bool_val(val):
    return isinstance(val, bool)


def validate_positive(n: float = None):
    if (isinstance(n, float) or isinstance(n, int)) and n > 0:
        return True
    return False


def validate_positive_increasing_integer_list(n: list):
    biggest=0
    for i in n:
        if not validate_positive_int(i) or i <= biggest:
            return False
        biggest = i
    return True


def validate_positive_decreasing_integer_list(n: list):
    if n is None:
        return True
    if isinstance(n, list):
        smallest = n[0]
        if len(n) > 1:
            for i in n[1:]:
                if not validate_positive(i) or i >= smallest:
                    return False
                smallest = i
    return True


def is_whole_num(x):
    return x - x // 1 == 0
