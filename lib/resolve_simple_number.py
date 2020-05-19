"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Union


def resolve_simple_number(
    hparams: Union[tf.contrib.training.HParams, dict], _name: str
) -> Union[int, float]:
    """Obtain the configured number value from the configuration file.

    Arguments:
        hparams {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters.
        _name {str} -- config key name.

    Returns:
        Union[int, float] -- The resolved number.
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

    if type(_result) is not int and type(_result) is not float:
        raise TypeError("hyper-parameter {} is not a number".format(_name))

    return _result
