"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf


def default_general_hparams() -> tf.contrib.training.HParams:
    """Returns a HParams object with general hyper parameter settings.

    Returns:
        tf.contrib.training.HParams -- An instance of Hparams
    """

    return tf.contrib.training.HParams(batch_size=256)


def join_hparams(*hparams: tuple) -> tf.contrib.training.HParams:
    """Combine two instances of tf.contrib.training.HParams

    Returns:
        tf.contrib.training.HParams -- The combined Hparams
    """
    params = {}
    for hp in hparams:
        params.update(hp.values())
    return tf.contrib.training.HParams(**params)
