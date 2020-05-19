"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from lib.resolve_learning_rate import resolve_lr
from typing import Union, Tuple


OPTIMIZER_NAMES = {
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
}


def resolve_optimizer(
    hparams: Union[tf.contrib.training.HParams, dict],
    global_step: Union[tf.Tensor, int] = None,
) -> Tuple[
    Union[
        tf.train.AdagradOptimizer,
        tf.train.AdamOptimizer,
        tf.train.RMSPropOptimizer,
        tf.train.GradientDescentOptimizer,
    ],
    Union[tf.Tensor, float],
]:
    """Obtain the specified optimizer.
    Apply learning rate scheduling when it is specified.

    Arguments:
        hparams {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters.

    Keyword Arguments:
        global_step {Union[tf.Tensor, int]} -- Global step to use for the decay computation. Must not be negative. (default: {None})

    Returns:
        Tuple[Union[tf.train.AdagradOptimizer,tf.train.AdamOptimizer,tf.train.RMSPropOptimizer,tf.train.GradientDescentOptimizer,],Union[tf.Tensor, float],] -- The optimizer
    """

    if not hasattr(hparams, "optimizer"):
        raise ValueError("hparams instance did not have optimizer information")

    if isinstance(hparams.optimizer, tf.train.Optimizer):
        return hparams.optimizer

    elif isinstance(hparams.optimizer, str):
        if hparams.optimizer not in OPTIMIZER_NAMES:
            raise ValueError("Unknown optimizer")
        optimizer_cls = OPTIMIZER_NAMES[hparams.optimizer]

        lr = resolve_lr(hparams, global_step)
        if lr is not None:
            tf.summary.scalar("learning_rate", lr)
            return optimizer_cls(lr), lr
        else:
            # Should never reach here
            raise RuntimeError("got None as learning rate value at runtime")
