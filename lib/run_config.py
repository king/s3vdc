"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf


def _session_config() -> tf.ConfigProto:
    """Constructs a session config specifying gpu memory usage.

    Returns:
        tf.ConfigProto -- session config.
    """

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return session_config


def default_run_config(
    model_dir: str,
    save_summary_steps: int = 100,
    save_checkpoints_mins: int = 5,
    keep_checkpoint_max: int = 5,
) -> tf.estimator.RunConfig:
    """Constructs a tf.contrib.learn.RunConfig instance with the specified model dir and default values.

    Arguments:
        model_dir {str} -- The model directory to save checkpoints, summary outputs etc.

    Keyword Arguments:
        save_summary_steps {int} -- save summary every x steps (default: {100})
        save_checkpoints_mins {int} -- save checkpoints every x steps (default: {5})
        keep_checkpoint_max {int} -- keep maximum x checkpoints (default: {5})

    Returns:
        tf.estimator.RunConfig -- The constructed RunConfig.
    """

    return tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=None,
        save_checkpoints_secs=save_checkpoints_mins * 60,  # seconds
        keep_checkpoint_max=keep_checkpoint_max,
        session_config=_session_config(),
    )
