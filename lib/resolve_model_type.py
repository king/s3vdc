"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""

import tensorflow as tf
from lib.user_custom_model import estimator_fn as user_custom_model
from typing import Callable, Tuple


# for this release, we only support custom model type
MODEL_TYPE_NAMES = {"userCustomModel": user_custom_model}


def resolve_model_type(
    model_type: str,
) -> Callable[
    [
        dict,
        dict,
        tf.contrib.training.HParams,
        str,
        list,
        list,
        list,
        tf.estimator.RunConfig,
    ],
    Tuple[dict, tf.Tensor, tf.Operation, dict, list, list],
]:
    """Obtained the model estimator function according to model type name

    Arguments:
        model_type {str} -- name of the model type

    Returns:
        Callable[[dict,dict,tf.contrib.training.HParams,str,list,list,list,tf.estimator.RunConfig,],Tuple[dict, tf.Tensor, tf.Operation, dict, list, list],] -- The estimator function.
    """

    if model_type not in MODEL_TYPE_NAMES:
        raise ValueError("Unknown modelType")

    return MODEL_TYPE_NAMES[model_type]
