"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import os
import multiprocessing
import tensorflow as tf
from lib.data_utils import DataUtils
from lib.tf_feature_columns import TfFeatureColumns
from lib.traindev_config_parser import TrainDevConfigParser
from typing import Union, Tuple


class DataSource:
    """
    A representation of a certain data source (train/eval/test)
    """

    max_allocatable_ram_bytes = (
        8 * 2 ** 10 * 2 ** 10 * 2 ** 10
    )  # 8 GB on a BASIC GPU tier

    def __init__(
        self,
        _tf_feature_cols: TfFeatureColumns,
        _ds_class: str,
        _dsutil: DataUtils = None,
        force_test: bool = False,
    ) -> None:
        """[summary]

        Arguments:
            _tf_feature_cols {TfFeatureColumns} -- input features
            _ds_class {str} -- dataset class

        Keyword Arguments:
            _dsutil {DataUtils} -- a reusable data utility if any (default: {None})
            force_test {bool} -- force a test when True (default: {False})
        """

        self.max_buffer_size = 1000000  # This is an empirical number yet to be adjusted according to per-sample size
        self.min_buffer_size = (
            10000  # This is the maximum number of samples allowed in each TFRecord file
        )

        self.declared_ds_class = _ds_class

        self._data_store = {"train": [], "eval": [], "test": []}
        if _dsutil is None:
            self._data_utils = DataUtils()
        else:
            self._data_utils = _dsutil

        if _ds_class not in set(self.data_store.keys()):
            raise TypeError("invalid data store class {}".format(_ds_class))
        self._tf_feature_cols = _tf_feature_cols
        # Load the data files for this data store
        if (
            self.add_datastore_entry(
                _entry_name=_ds_class, _path=self.train_dev_conf.data_source[_ds_class]
            )
            == 0
        ):
            tf.logging.warn("no data files found for {}".format(_ds_class))
        else:
            tf.logging.info(
                "{} dataset contains files: {}".format(
                    _ds_class, str(self.ds_file_list)
                )
            )

        self.num_cpus = multiprocessing.cpu_count()
        tf.logging.info(
            "{} CPUs discovered on the hosting VM".format(str(self.num_cpus))
        )

        # if force_test is true, it will treat all dataset types as test dataset (epoch = 1)
        self.force_test = force_test

        # adjust max_buffer_size
        max_buffer_size_candidates = [self.max_buffer_size]
        # consider number of training samples
        data_source_conf = _tf_feature_cols.train_dev_config_parser.data_source
        if "numTrain" in data_source_conf:
            max_buffer_size_candidates.append(int(data_source_conf["numTrain"]))
        # consider max ram
        max_buffer_size_candidates.append(
            int(
                self.max_allocatable_ram_bytes
                / _tf_feature_cols.estimated_bytes_per_sample
            )
        )
        # pick min from candidates
        self.max_buffer_size = min(max_buffer_size_candidates)
        tf.logging.info(
            "set max_buffer_size={} from candidates {}.".format(
                str(self.max_buffer_size), str(max_buffer_size_candidates)
            )
        )
        # adjust min_buffer_size
        if self.min_buffer_size > self.max_buffer_size:
            self.min_buffer_size = self.max_buffer_size
        tf.logging.info("set min_buffer_size={}".format(str(self.min_buffer_size)))

    @property
    def data_utils(self) -> DataUtils:
        """Get an instance of DataUtils

        Returns:
            DataUtils -- an instance of DataUtils
        """
        return self._data_utils

    @property
    def ds_class(self) -> str:
        """Get the dataset class

        Returns:
            str -- one of {train, eval, test}
        """
        for _class in self.data_store:
            if len(self.data_store[_class]) > 0:
                return _class
        return None

    @property
    def is_empty(self) -> bool:
        """Is the dataset empty?

        Returns:
            bool -- True when the dataset is empty
        """
        if self.ds_class is None:
            return True
        else:
            return False

    @property
    def ds_file_list(self) -> list:
        """Get the list of dataset files

        Returns:
            list -- a list of dataset files
        """
        return self.data_store[self.ds_class]

    @property
    def tf_feature_cols(self) -> TfFeatureColumns:
        """Get the object of TfFeatureColumns

        Returns:
            TfFeatureColumns -- the object of TfFeatureColumns
        """
        return self._tf_feature_cols

    @property
    def train_dev_conf(self) -> TrainDevConfigParser:
        """Get the object of TrainDevConfigParser

        Returns:
            TrainDevConfigParser -- the object of TrainDevConfigParser
        """
        return self._tf_feature_cols.train_dev_config_parser

    @property
    def data_store(self) -> dict:
        """Get the data store

        Returns:
            dict -- the dict of data store
        """
        return self._data_store

    @staticmethod
    def try_sparse2dense_tensor(_tensor: tf.Tensor) -> tf.Tensor:
        """Try to convert a sparse Tensor to a dense one

        Arguments:
            _tensor {tf.Tensor} -- input tensor

        Returns:
            tf.Tensor -- output dense tensor
        """
        if isinstance(_tensor, tf.SparseTensor):
            if _tensor.values.dtype == "string":
                # _tensor = tf.sparse_tensor_to_dense(_tensor, default_value='__unknown__')
                # _tensor = tf.convert_to_tensor(_tensor)
                _tensor = tf.sparse.to_dense(
                    sp_input=_tensor, default_value="__unknown__"
                )
            else:
                _tensor = tf.sparse_tensor_to_dense(_tensor)
        return _tensor

    def _add_local_datastore_entries(self, _entry_name: str, _path: str) -> int:
        """Find and add all physical dataset files

        Arguments:
            _entry_name {str} -- dataset class name
            _path {str} -- path of the dataset

        Returns:
            int -- number of files added
        """

        _file_list = []
        for f in os.listdir(_path):
            _f = os.path.join(_path, f)
            if not f.startswith(".") and os.path.isfile(_f):
                _file_list.append(_f)
        self.data_store[_entry_name] = _file_list
        return len(_file_list)

    def add_datastore_entry(self, _entry_name: str, _path: str) -> int:
        """Add all tfrecord files under path _path to _entry_name in _data_store.

        Arguments:
            _entry_name {str} -- dataset class name
            _path {str} -- path of the dataset

        Returns:
            int -- number of files added
        """

        if self.ds_class is not None:
            raise AssertionError("the data source has already been initialized")
        if _entry_name in set(self.data_store.keys()):
            self._add_local_datastore_entries(_entry_name=_entry_name, _path=_path)
        else:
            raise KeyError("invalid key {} in data_store".format(_entry_name))
        return len(self.data_store[_entry_name])

    def _parse_function(
        self, serialized_batch: Union[list, tf.Tensor]
    ) -> Tuple[dict, dict]:
        """Generate parsed features and labels from a serialized batch of samples

        Arguments:
            serialized_batch {Union[list, tf.Tensor]} -- see return value of tf.compat.v1.train.batch

        Returns:
            Tuple[dict, dict] -- dict of parsed features and labels
        """
        _parse_example_spec = tf.feature_column.make_parse_example_spec(
            self.tf_feature_cols.all_feature_columns
        )
        parsed_examples = tf.parse_example(serialized_batch, _parse_example_spec)

        labels = {}
        _label_column_keys = self.tf_feature_cols.label_column_names
        if len(_label_column_keys) > 0:
            for cur_label in _label_column_keys:
                labels[cur_label] = parsed_examples.pop(cur_label)

        return parsed_examples, labels

    def input_fn(
        self, batch_size: int = 512, compression: str = "GZIP", max_epochs: int = None
    ):
        """Input function for this data set. Note that tf.estimator.Estimator instances expect a 0-ary function
        meaning if any of the arguments for this function should be set, a lambda or partial could be passed
        instead.

        Keyword Arguments:
            batch_size {int} -- max number of samples per batch. (default: {512})
            compression {str} -- the type of compression (string); None for no compression. (default: {"GZIP"})
            max_epochs {int} -- the maxinum number of epochs for training (reserved). (default: {None})

        Returns:
            [type] -- A nested structure of tf.Tensor objects (features dict, labels tensor).
        """

        with tf.variable_scope("Data_source"):

            # training needs full-reshuffle
            if self.ds_class == "train" and not self.force_test:

                _num_dsfiles = len(self.ds_file_list)
                parallelism = self.num_cpus
                if _num_dsfiles < self.num_cpus:
                    parallelism = _num_dsfiles

                dataset = tf.data.Dataset.from_tensor_slices(self.ds_file_list)
                dataset = dataset.shuffle(_num_dsfiles)

                dataset = dataset.interleave(
                    lambda filename: tf.data.TFRecordDataset(
                        filenames=filename, compression_type=compression
                    ),
                    cycle_length=parallelism,
                )

                # TODO: better way of determining buffer_size
                _bufsize = batch_size * 10 ** 2
                if _bufsize < self.min_buffer_size:
                    _bufsize = self.min_buffer_size
                elif _bufsize > self.max_buffer_size:
                    _bufsize = self.max_buffer_size
                tf.logging.info("set buffer size to {}".format(str(_bufsize)))

                # This order is recommended to maintain the boundary of epochs
                # https://www.tensorflow.org/performance/datasets_performance#repeat_and_shuffle
                # dataset = dataset.shuffle(buffer_size=_bufsize)
                # dataset = dataset.repeat()
                dataset = dataset.apply(
                    tf.data.experimental.shuffle_and_repeat(
                        buffer_size=_bufsize, count=max_epochs
                    )
                )

            else:
                dataset = tf.data.TFRecordDataset(
                    self.ds_file_list, compression_type=compression
                )
                # dataset = dataset.repeat(1)

            # TODO: the following two lines might be able to be merged into one
            dataset = dataset.batch(batch_size=batch_size)
            dataset = dataset.map(
                self._parse_function, num_parallel_calls=self.num_cpus
            )

            iterator = dataset.make_one_shot_iterator()

            return iterator.get_next()
