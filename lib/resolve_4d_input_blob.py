"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from tensorflow import feature_column as tfc
from lib.check_array_feature import check_array_feature
from lib.find_feature_column import find_feature_column
from lib.resolve_simple_number_array import resolve_simple_number_array


def resolve_4d_input_blob(
    hparams: tf.contrib.training.HParams,
    runtime_bs: tf.Tensor,
    features: dict,
    feature_columns: list,
    info_log: bool = False,
) -> tf.Tensor:
    """Convert a dict feature input to a 4D input cube with dimension (NHWC).
    this function is experimental.

    Arguments:
        hparams {tf.contrib.training.HParams} -- hyper parameters
        runtime_bs {tf.Tensor} -- the batch_size in runtime
        features {dict} -- the dict of feature
        feature_columns {list} -- the list of feature columns

    Keyword Arguments:
        info_log {bool} -- True to enable debugging info logging (default: {False})

    Returns:
        tf.Tensor -- The created 4D input Tensor
    """

    # initialize vars
    frame_shape = resolve_simple_number_array(hparams, "frameShape")

    batch_input_shape_tensor = tf.convert_to_tensor([runtime_bs] + frame_shape)

    padding = resolve_simple_number_array(hparams, "padding")

    # Process time-series and non-time-series features one by one
    feature_list = []
    for key in sorted(features):

        cur_feature = features[key]
        is_array, _ = check_array_feature(cur_feature)

        # build ts feature planes
        if is_array:
            # padding
            if sum(padding) > 0:
                padding_tensor = tf.constant([[0, 0], padding])
                cur_feature = tf.pad(
                    cur_feature, padding_tensor, mode="CONSTANT", constant_values=0
                )
            # reshape
            cur_feature = tf.reshape(cur_feature, batch_input_shape_tensor)
            # cast to float
            if cur_feature.dtype != tf.float32:
                cur_feature = tf.cast(cur_feature, dtype=tf.float32)
            # add to list with added channel dim (NHWC)
            feature_list.append(cur_feature[:, :, :, tf.newaxis])
            # log ts feature
            if info_log:
                tf.logging.info("{}: {}".format(key, cur_feature))

        # build non-ts feature planes (Numerical Features)
        # note that we treat SparseTensor and Tensor with dtype=string as categorical features
        elif type(cur_feature) is tf.Tensor and cur_feature.dtype.name != "string":
            # tiling
            cur_feature = tf.tile(
                cur_feature[:, tf.newaxis], [1, frame_shape[0] * frame_shape[1]]
            )
            # reshape
            cur_feature = tf.reshape(cur_feature, batch_input_shape_tensor)
            # cast to float
            if cur_feature.dtype != tf.float32:
                cur_feature = tf.cast(cur_feature, dtype=tf.float32)
            # add to list with added channel dim (NHWC)
            feature_list.append(cur_feature[:, :, :, tf.newaxis])
            # log numerical feature
            if info_log:
                tf.logging.info("{}: {}".format(key, cur_feature))

        # build non-ts feature planes (Categorical Features)
        else:
            cur_feature = tfc.input_layer(
                {key: cur_feature}, find_feature_column(key, feature_columns)
            )
            # padding
            cur_feature = tf.tile(
                cur_feature[:, :, tf.newaxis], [1, 1, frame_shape[0] * frame_shape[1]]
            )
            # split
            cur_features = tf.split(
                cur_feature, axis=1, num_or_size_splits=cur_feature.shape[1]
            )
            # process each feature plane
            for entry in cur_features:
                # reshape
                entry = tf.reshape(entry, batch_input_shape_tensor)
                # cast to float
                if entry.dtype != tf.float32:
                    entry = tf.cast(entry, dtype=tf.float32)
                # add to list with added channel dim (NHWC)
                feature_list.append(entry[:, :, :, tf.newaxis])
                # log categorical feature plane
                if info_log:
                    tf.logging.info("{}: {}".format(key, entry))

    # channel stacking
    data = tf.concat(feature_list, -1)

    # interpolation
    interp = resolve_simple_number_array(hparams, "interp")
    if interp is not None and interp != frame_shape:
        data = tf.image.resize_images(data, tf.convert_to_tensor(interp))

    return data
