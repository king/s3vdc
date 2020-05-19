"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Union


ACTIVATION_FN_NAMES = {"RELU": tf.nn.relu, "ELU": tf.nn.elu}


def resolve_activation_fn(
    hparams: Union[tf.contrib.training.HParams, dict],
) -> Union[tf.nn.relu, tf.nn.elu]:
    """Resolve network activation function according to hyper-parameters

    Arguments:
        hparams {Union[tf.contrib.training.HParams, dict]} -- hyper-parameters

    Returns:
        Union[tf.nn.relu, tf.nn.elu] -- The resolved activation function
    """

    if not hasattr(hparams, "activationFn"):
        raise ValueError("hparams instance did not have activationFn information")

    if hparams.activationFn not in ACTIVATION_FN_NAMES:
        raise ValueError("Unknown optimizer")

    return ACTIVATION_FN_NAMES[hparams.activationFn]
