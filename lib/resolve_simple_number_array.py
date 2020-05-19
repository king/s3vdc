"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Union, List, Type


def resolve_simple_number_array(
    hparams: Union[tf.contrib.training.HParams, dict],
    _name: str,
    _num_type: Type = int,
) -> List[Union[int, float]]:
    """Obtain the config in the format of number array.

    Arguments:
        hparams {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters.
        _name {str} -- key name of this config.

    Keyword Arguments:
        _num_type {Type} -- The type of the number in the array. (default: {int})

    Returns:
        List[Union[int, float]] -- a number list, the elements of which is of type _num_type
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

    if type(_result) is not list:
        raise TypeError("hyper-parameter {} is not an array".format(_name))

    for entry in _result:
        if type(entry) is not _num_type:
            raise TypeError(
                "{} in {} array is not of type {}".format(
                    str(entry), _name, str(_num_type)
                )
            )

    return _result
