"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""

import tensorflow as tf


def find_feature_column(name: str, feature_columns: list) -> tf.feature_column:
    """Find the feature column with a specific name

    Arguments:
        name {str} -- the name to search for
        feature_columns {list} -- the list of feature columns to search from

    Returns:
        tf.feature_column -- the first found feature column
    """

    for e in feature_columns:
        if e.name.find(name) == 0:
            return e
