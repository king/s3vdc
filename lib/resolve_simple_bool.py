"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Union


def resolve_simple_bool(
    hparams: Union[tf.contrib.training.HParams, dict], _name: str
) -> bool:
    """Obtain a bool value with a key name from config.

    Arguments:
        hparams {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters.
        _name {str} -- Key name.

    Returns:
        bool -- The actual value.
    """

    _result = None

    if isinstance(hparams, dict):
        if _name not in hparams:
            raise ValueError("{} missing from hyper-parameters".format(_name))
        _result = hparams[_name]
    else:
        if not hasattr(hparams, _name):
            raise ValueError("{} missing from hyper-parameters".format(_name))
        _result = getattr(hparams, _name)

    if type(_result) is not bool:
        raise TypeError("hyper-parameter {} is not a bool".format(_name))

    return _result
