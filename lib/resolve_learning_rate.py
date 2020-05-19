"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Union


def resolve_lr_schedule_constant(
    lr: Union[tf.Tensor, float], *dummys
) -> Union[tf.Tensor, float]:
    """A utility function for simply returning the learning rate

    Arguments:
        lr {Union[tf.Tensor, float]} -- learning rate
        dummys {Any type} -- dummy parameters in order to be compatible with other lr resolvers

    Returns:
        Union[tf.Tensor, float] -- the resolved learning rate
    """

    if type(lr) not in {float, tf.Tensor}:
        raise TypeError("expect learning rate to be float, but got {}".format(type(lr)))

    return lr


def resolve_lr_schedule_exponential(
    lr: Union[tf.Tensor, float],
    global_step: Union[tf.Tensor, int],
    hparams: Union[tf.contrib.training.HParams, dict],
) -> tf.Tensor:
    """Applies exponential decay to the learning rate.

    Arguments:
        lr {Union[tf.Tensor, float]} -- The initial learning rate.
        global_step {Union[tf.Tensor, int]} -- Global step to use for the decay computation. Must not be negative.
        hparams {Union[tf.contrib.training.HParams, dict]} -- hyper parameers

    Returns:
        tf.Tensor -- The decayed learning rate.
    """

    if not hasattr(hparams, "lrDecayParams"):
        raise AttributeError(
            "missing parameter lrDecayParams for exponential learning rate decay"
        )

    lr_decay_params = hparams.lrDecayParams

    n_lr_decay_params = len(lr_decay_params)
    if n_lr_decay_params != 2:
        raise AttributeError(
            "expect exactly 2 parameters for exponential learning rate decay, but got {}".format(
                str(n_lr_decay_params)
            )
        )

    type_lr_decay_params_0 = type(lr_decay_params[0])
    if type_lr_decay_params_0 is not int:
        raise AttributeError(
            "expect first parameter for exponential learning rate decay to be int, got {}".format(
                str(type_lr_decay_params_0)
            )
        )

    type_lr_decay_params_1 = type(lr_decay_params[1])
    if type_lr_decay_params_1 is not float:
        raise AttributeError(
            "expect first parameter for exponential learning rate decay to be float, got {}".format(
                str(type_lr_decay_params_1)
            )
        )

    return tf.train.exponential_decay(
        lr, global_step, lr_decay_params[0], lr_decay_params[1]
    )


LEARNING_RATE_SCHEDULE_TYPES = {
    "Exponential": resolve_lr_schedule_exponential,
    "Constant": resolve_lr_schedule_constant,
}


def resolve_lr(
    hparams: Union[tf.contrib.training.HParams, dict],
    global_step: Union[tf.Tensor, int] = None,
) -> tf.Tensor:
    """Obtain the final learning rate.
    Apply learning rate scheduling if configured.
    Apply minimum learning rate if configured.

    Arguments:
        hparams {Union[tf.contrib.training.HParams, dict]} -- hyper parameters.

    Keyword Arguments:
        global_step {Union[tf.Tensor, int]} -- Global step to use for the decay computation. Must not be negative. (default: {None})

    Returns:
        tf.Tensor -- The final learning rate.
    """

    # The default learning rate will be set to 0.01
    if not hasattr(hparams, "learningRate"):
        return 0.01

    lr = hparams.learningRate
    tf.logging.info("set initial learning rate to {}".format(lr))

    if not hasattr(hparams, "lrDecayType"):
        return lr

    if hparams.lrDecayType not in LEARNING_RATE_SCHEDULE_TYPES:
        return lr

    result_lr = LEARNING_RATE_SCHEDULE_TYPES[hparams.lrDecayType](
        lr, global_step, hparams
    )

    if hasattr(hparams, "minLearningRate"):
        min_lr = hparams.minLearningRate
        if type(min_lr) is float:
            result_lr = tf.maximum(result_lr, min_lr)
            tf.logging.info("set minLearningRate={}".format(min_lr))
        else:
            tf.logging.info(
                "expect float number for minLearningRate, got {}".format(str(min_lr))
            )

    return result_lr
