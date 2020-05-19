"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
import sys
from tensorflow import logging as tfl
from lib.traindev_config_parser import TrainDevConfigParser
from typing import Union


class TfFeatureColumns:
    """
    Representation of tf.feature_columns list built from TrainDev configuration
    """

    # TODO: smarter way of determine max_embedding_size
    max_embedding_size = 10000

    def __init__(self, _td_cp: TrainDevConfigParser) -> None:
        """Initialize a TfFeatureColumns.

        Arguments:
            _td_cp {TrainDevConfigParser} -- An instance of TrainDevConfigParser.
        """
        self._td_conf_parser = _td_cp

        # _tf_export_cols contains all columns to be exported to the serving model
        self._tf_export_cols = []
        # _tf_export_cols contains the columns to be excluded from the serving model
        self._tf_no_export_cols = []

        # construct the lists of tf.feature_columns for features only
        self._tf_feature_cols = []
        for _name in self._td_conf_parser.feature_column_names:
            _tmp_tf_feature_column = self.create_tf_feature_column(_name)
            self._tf_feature_cols.append(_tmp_tf_feature_column)

            # fill _tf_export_cols and _tf_no_export_cols
            if _name in self._td_conf_parser.no_export_column_names:
                self._tf_no_export_cols.append(_tmp_tf_feature_column)
                tfl.info(
                    "found control feature {}, and will exclude it from the exported model.".format(
                        _name
                    )
                )
            else:
                self._tf_export_cols.append(_tmp_tf_feature_column)

        # construct the lists of tf.feature_columns for labels only
        self._label_tf_feature_cols = []
        for _name in self._td_conf_parser.label_column_names:
            self._label_tf_feature_cols.append(self.create_tf_feature_column(_name))

        # construct the lists of tf.feature_columns for IDs only
        self._id_tf_feature_cols = []
        for _name in self._td_conf_parser.id_column_names:
            self._id_tf_feature_cols.append(self.create_tf_feature_column(_name))
            self._tf_export_cols.append(self.create_tf_feature_column(_name))

        # and at the same time, calculate estimated_bytes_per_sample
        self._estimated_bytes_per_sample = 0
        for _name in self._td_conf_parser.column_names:
            self._estimated_bytes_per_sample += self.estimate_bytes_for_fc(_name)

    @property
    def train_dev_config_parser(self) -> TrainDevConfigParser:
        """Return the initialized object of ConfigParser

        Returns:
            TrainDevConfigParser -- An instance of ConfigParser
        """
        return self._td_conf_parser

    @property
    def all_feature_columns(self) -> list:
        """Return the constructed tf.feature_columns list.

        Returns:
            list -- A list of tf.feature_column
        """
        return (
            self._tf_feature_cols
            + self._label_tf_feature_cols
            + self._id_tf_feature_cols
        )

    @property
    def export_feature_columns(self) -> list:
        """Return the constructed tf.feature_columns list that will be included in the exported serving model.

        Returns:
            list -- A list of tf.feature_columns
        """
        return self._tf_export_cols

    @property
    def no_export_feature_columns(self) -> list:
        """Return the constructed tf.feature_columns list that will be excluded from the exported serving model.

        Returns:
            list -- A list of tf.feature_columns
        """
        return self._tf_no_export_cols

    @property
    def id_feature_columns(self) -> list:
        """Get the lists of tf.feature_columns for IDs only.

        Returns:
            list -- A list of tf.feature_columns
        """
        return self._id_tf_feature_cols

    @property
    def label_feature_columns(self) -> list:
        """Get the lists of tf.feature_columns for labels only.

        Returns:
            list -- A list of tf.feature_columns
        """
        return self._label_tf_feature_cols

    @property
    def only_feature_columns(self) -> list:
        """Get the lists of tf.feature_columns for features only.

        Returns:
            list -- A list of tf.feature_columns
        """
        return self._tf_feature_cols

    @property
    def label_column_names(self) -> list:
        """Return set of names of label columns.

        Returns:
            list -- A sorted list of label column names.
        """
        return self._td_conf_parser.label_column_names

    @property
    def id_column_names(self) -> list:
        """Return set of names of ID columns.

        Returns:
            list -- A sorted list of ID columns.
        """
        return self._td_conf_parser.id_column_names

    @property
    def estimated_bytes_per_sample(self) -> Union[int, float]:
        """Return the estimated bytes for every sample.

        Returns:
            int -- The number of estimated bytes.
        """
        return self._estimated_bytes_per_sample

    @staticmethod
    def feature_column_key(_tf_fcobj: tf.feature_column) -> str:
        """Return the key of the corresponding feature_column.

        Arguments:
            _tf_fcobj {tf.feature_column} -- The target feature_column object.

        Returns:
            str -- The found key.
        """
        _key = None
        if hasattr(_tf_fcobj, "key"):
            _key = _tf_fcobj.key
        elif hasattr(_tf_fcobj, "categorical_column") and hasattr(
            _tf_fcobj.categorical_column, "key"
        ):
            _key = _tf_fcobj.categorical_column.key
        else:
            raise KeyError(
                "no key found for feature_column object {}".format(str(_tf_fcobj))
            )
        return _key

    def estimate_bytes_for_fc(self, _name: str) -> Union[int, float]:
        """Estimate the number of bytes for feature columns _name.

        Arguments:
            _name {str} -- The name of the target feature column.

        Returns:
            Union[int, float] -- The estimated number of bytes.
        """
        _fc_type = self._td_conf_parser.fc_type(_name)
        _fc_params = self._td_conf_parser.fc_params(_name)
        _dtype = self._td_conf_parser.dtype(_name)

        _shape = self._td_conf_parser.shape(_name)
        n_shape = 1 if len(_shape) < 1 else _shape[0]

        # fcNumeric
        if _fc_type == "fcNumeric":
            if _dtype is tf.float32:
                return 4 * n_shape
            else:
                return 8 * n_shape

        # fcVocabIndicator
        elif _fc_type == "fcVocabIndicator":
            if _dtype is tf.string and "vocabulary" in _fc_params:
                vocab = _fc_params["vocabulary"]
                vocab_bytes = 0
                for entry in vocab:
                    vocab_bytes += sys.getsizeof(entry.encode("utf-8"))
                return vocab_bytes / len(vocab)
            else:
                raise AttributeError(
                    "illegal combination for {}: dtype={}, vocabulary={}".format(
                        _name, str(_dtype), str("vocabulary" in _fc_params)
                    )
                )

        # fcHashEmbedding
        elif _fc_type == "fcHashEmbedding":
            return 50  # A better estimation need information from EDA. Use TFT is probably an overkill.

        else:
            raise ValueError(
                "unsupported fcType {} for column {}".format(_fc_type, _name)
            )

    def create_tf_feature_column(self, _name: str) -> tf.feature_column:
        """Return a certain type of feature column according to the configuration of column _name.

        Arguments:
            _name {str} -- The name (string) of the target column.

        Returns:
            tf.feature_column -- An instance of a class in tf.feature_column.
        """
        _tf_feature_column = None
        _fc_type = self._td_conf_parser.fc_type(_name)
        _fc_params = self._td_conf_parser.fc_params(_name)
        _shape = self._td_conf_parser.shape(_name)
        _dtype = self._td_conf_parser.dtype(_name)

        # fcNumeric
        if _fc_type == "fcNumeric":
            _tf_feature_column = tf.feature_column.numeric_column(
                _name, shape=_shape, dtype=_dtype
            )

        # fcVocabIndicator
        elif _fc_type == "fcVocabIndicator":
            if "vocabulary" in _fc_params:
                _tf_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
                    _name, vocabulary_list=_fc_params["vocabulary"]
                )
            else:
                raise AttributeError(
                    "missing vocabulary attribute for column {}".format(_name)
                )
            _tf_feature_column = tf.feature_column.indicator_column(_tf_feature_column)

        # fcHashEmbedding
        elif _fc_type == "fcHashEmbedding":
            if "hashSize" in _fc_params:
                hashSize = _fc_params["hashSize"]
                _tf_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
                    _name,
                    hash_bucket_size=(
                        hashSize
                        if hashSize <= self.max_embedding_size
                        else self.max_embedding_size
                    ),
                    dtype=_dtype,
                )
            else:
                raise AttributeError("missing hashSize for feature {}".format(_name))
            if "embeddingSize" in _fc_params:
                _dimension = _fc_params["embeddingSize"]
                if _name == "core_user_id":
                    _dimension = 64
                _tf_feature_column = tf.feature_column.embedding_column(
                    _tf_feature_column, dimension=_dimension
                )
            else:
                raise AttributeError(
                    "missing embeddingSize for feature {}".format(_name)
                )

        else:
            raise ValueError("unsupported fcType for column {}".format(_name))

        return _tf_feature_column
