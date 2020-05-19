"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Tuple


def check_array_feature(tensor: tf.Tensor) -> Tuple[bool, int]:
    """For a feature, check if it is a time series (array) feature.
    If it is a time series feature, also return the length.

    Arguments:
        tensor {tf.Tensor} -- A feature represented by a tensor

    Returns:
        Tuple[bool, int]
            - is_array {bool} - is array tensor
            - array_length {int} - number of elements in the array
    """

    if type(tensor) is not tf.Tensor:
        return False, None

    if not hasattr(tensor, "shape"):
        return False, None

    shape = tensor.shape
    if shape[-1] > 0:
        return True, int(shape[-1])

    return False, None
