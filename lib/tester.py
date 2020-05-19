"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
import os
import io
import datetime
import json
import csv
from lib.data_source import DataSource
from lib.run_config import default_run_config
from lib.hyper_parameters import join_hparams, default_general_hparams
from collections import namedtuple
from typing import Callable, Union, Tuple

TestResults = namedtuple("TestResult", ["estimator", "metrics"])


class Tester:
    """
    Implementation of a Tester class
    """

    prediction_file_format = {"CSV": ".csv", "TEXT": ".txt", "JSON": ".json"}

    default_num_lines_per_file = 100000

    def __init__(
        self,
        model_fn: Callable[
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
        ],
        model_dir: str,
        test_data_source: DataSource,
        hyper_params: Union[tf.contrib.training.HParams, dict] = None,
        hooks: list = None,
        date_time_str: str = None,
    ) -> None:
        """Initialize a Tester

        Arguments:
            model_fn {Callable[[dict,dict,tf.contrib.training.HParams,str,list,list,list,tf.estimator.RunConfig,],Tuple[dict, tf.Tensor, tf.Operation, dict, list, list],]} -- A model function.
            model_dir {str} -- The model directory.
            test_data_source {DataSource} -- The DataSource object of test dataset.

        Keyword Arguments:
            hyper_params {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters. (default: {None})
            hooks {list} -- A list of custom hooks. (default: {None})
            date_time_str {str} -- A date and time string in the format of "yyyymmdd_hhmmss" (default: {None})
        """

        # run_config
        # TODO: make run_config configurable
        self.model_dir = model_dir
        self.run_config = default_run_config(self.model_dir)

        # date and time string
        self.date_time_str = date_time_str

        # default test/prediction output path
        self.prediction_output_path = ""
        self.test_metric_output_path = ""
        if date_time_str is None:
            tf.logging.warning(
                "missing date and time info as input params for test/prediction job"
            )
            self.prediction_output_path = os.path.join(
                model_dir,
                "prediction_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            self.test_metric_output_path = os.path.join(
                model_dir,
                "test_metric_{}.json".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
            )
        else:
            self.prediction_output_path = os.path.join(
                model_dir, "prediction_" + date_time_str
            )
            self.test_metric_output_path = os.path.join(
                model_dir, "test_metric_{}.json".format(date_time_str)
            )

        # hyper_params
        if hyper_params is None:
            hyper_params = {}
        hyper_params = tf.contrib.training.HParams(**hyper_params)
        self.hyper_params = join_hparams(default_general_hparams(), hyper_params)

        # TODO: extend the functions via different hooks
        self.hooks = hooks

        # data_sources
        self.test_data_source = test_data_source

        # model_fn
        self.model_fn = model_fn

        # feature columns
        self.only_feature_columns = (
            test_data_source.tf_feature_cols.only_feature_columns
        )
        # self.id_feature_columns = test_data_source.tf_feature_cols.id_feature_columns
        # self.label_feature_columns = test_data_source.tf_feature_cols.label_feature_columns

        # estimator
        self.estimator = self.model_fn(
            self.only_feature_columns,
            self.run_config,
            self.hyper_params,
            test_data_source.tf_feature_cols,
        )

    def run_predictions(
        self,
        file_format: str = None,
        num_lines_per_file: int = default_num_lines_per_file,
        num_samples: int = 0,
        content_encoding: str = None,
    ) -> str:
        """Run prediction

        Keyword Arguments:
            file_format {str} -- The output file format, see keys in prediction_file_format. (default: {None})
            num_lines_per_file {int} -- The number of lines per file. (default: {default_num_lines_per_file})
            num_samples {int} -- The number of samples (usually quite small) to be included in the sample.json. (default: {0})
            content_encoding {str} -- The encoding of the file content, that can be None or "GZIP". (default: {None})

        Returns:
            str -- The output path of the prediction results.
        """

        tf.logging.info("Getting Predictions ...")
        file_format = file_format.upper()

        pred_result = self.estimator.predict(input_fn=self.test_data_source.input_fn)

        if file_format not in set(self.prediction_file_format.keys()):
            tf.logging.warning(
                "unsupported prediction output file type {}".format(file_format)
            )
            return pred_result

        if file_format == "CSV":
            self.write_predictions_to_csv_files(
                pred_result, num_lines_per_file, num_samples, content_encoding
            )
        elif file_format == "JSON":
            self.write_predictions_to_json_files(
                pred_result, num_lines_per_file, num_samples, content_encoding
            )
        elif file_format == "TEXT":
            self.write_predictions_to_txt_files(
                pred_result, num_lines_per_file, num_samples, content_encoding
            )

        return self.prediction_output_path

    @staticmethod
    def list_of_dicts_to_csv_str(list_of_dicts: list) -> str:
        """Convert a list of dict to a CSV string

        Arguments:
            list_of_dicts {list} -- A list of dict

        Returns:
            str -- The converted CSV string.
        """

        if len(list_of_dicts) < 1:
            raise ValueError("at least one dict should exist in the list")

        header = list(list_of_dicts[0].keys())

        f = io.StringIO("io_list_of_dicts_to_csv_str")
        csv_writer = csv.DictWriter(f, header, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writeheader()
        for entry in list_of_dicts:
            csv_writer.writerow(entry)

        result = f.getvalue()
        f.close()
        return result

    def prediction_record_to_dict(
        self, record: dict, idx: int = None, id_keys: list = None
    ) -> Tuple[dict, dict]:
        """Convert prediction record to a tuple of dict

        Arguments:
            record {dict} -- The prediction record.

        Keyword Arguments:
            idx {int} -- Current line index. (default: {None})
            id_keys {list} -- A list of feature names that are treated as IDs. (default: {None})

        Returns:
            Tuple[dict, dict] -- A tuple of ID and prediction result.
        """

        if id_keys is None:
            id_keys = self.test_data_source.train_dev_conf.id_column_names

        pred_result = {}
        ids = {}

        if idx is not None:
            ids["index"] = idx

        for _key in record:
            tmp_list = []
            try:
                for _element in record[_key]:
                    if type(_element) is bytes:
                        tmp_list.append(_element.decode("utf-8"))
                    else:
                        tmp_list.append(_element.item())
            except TypeError:
                tmp_list.append(record[_key].item())
            except AttributeError:
                tmp_list.append(record[_key].decode("utf-8"))

            # un-nest list in case of only one element
            if len(tmp_list) == 1:
                tmp_list = tmp_list[0]

            # return ids and predictions separately
            if _key in id_keys:
                ids[_key] = tmp_list
            else:
                pred_result[_key] = tmp_list

        return ids, pred_result

    def write_once(
        self,
        last_idx: int,
        idx: int,
        file_suffix: str,
        num_samples: int,
        num_records_per_file: int,
        prediction_cache: list,
        content_encoding: str = None,
        content_type: str = "JSON",
    ) -> None:
        """Write a chunk of prediction records into a file.

        Arguments:
            last_idx {int} -- The beninning index.
            idx {int} -- The ending index.
            file_suffix {str} -- Suffix of the output file.
            num_samples {int} -- The number of prediction records in the sample.json file.
            num_records_per_file {int} -- The number of prediction records in each output file.
            prediction_cache {list} -- A list representing the prediction cache.

        Keyword Arguments:
            content_encoding {str} -- The encoding of the file content, that can be None or "GZIP". (default: {None})
            content_type {str} -- The output file format, see keys in prediction_file_format. (default: {"JSON"})
        """

        if content_type not in set(self.prediction_file_format.keys()):
            raise TypeError(
                "content_type {} not supported in write_once".format(content_type)
            )

        content = ""
        if content_type == "CSV":
            content = self.list_of_dicts_to_csv_str(prediction_cache)
        elif content_type == "JSON":
            for entry in prediction_cache:
                content += "{}\n".format(json.dumps(entry))
        elif content_type == "TEXT":
            for entry in prediction_cache:
                content += "{}\n".format(str(entry))

        _file_path = os.path.join(
            self.prediction_output_path,
            "part_{}-{}{}".format(str(last_idx), str(idx), file_suffix),
        )
        self.test_data_source.data_utils.write_to_file(
            content, _file_path, content_encoding
        )

        # output a json sample file (always in json format) if required
        if num_samples > 0 and idx / num_records_per_file <= 1:
            try:
                _file_path = os.path.join(self.prediction_output_path, "sample.json")
                self.test_data_source.data_utils.write_to_file(
                    json.dumps(prediction_cache[:num_samples]), _file_path
                )
            except TypeError:
                _file_path = os.path.join(self.prediction_output_path, "sample.txt")
                self.test_data_source.data_utils.write_to_file(
                    str(prediction_cache[:num_samples]), _file_path
                )
            except Exception as e:
                raise RuntimeError("Tester.write_once:\n{}".format(e.__str__))

    def write_predictions_to_json_files(
        self,
        prediction_result_generator: list,
        num_records_per_file: int = default_num_lines_per_file,
        num_samples: int = 0,
        content_encoding: str = None,
    ) -> None:
        """Write predictions into a JSON file.

        Arguments:
            prediction_result_generator {list} -- A yield of prediction result.

        Keyword Arguments:
            num_records_per_file {int} -- The number of records in each output file. (default: {default_num_lines_per_file})
            num_samples {int} -- the number of samples to be included in the sample.json; this int number shall be smaller than num_records_per_file (default: {0})
            content_encoding {str} -- The encoding of the file content, that can be None or "GZIP". (default: {None})
        """

        prediction_cache = []
        id_keys = self.test_data_source.train_dev_conf.id_column_names
        file_suffix = self.prediction_file_format["JSON"]
        last_idx, idx = 0, 0

        for idx, entry in enumerate(prediction_result_generator, start=0):

            # transform and split raw prediction record
            id_dict, pred_dict = self.prediction_record_to_dict(
                record=entry, idx=idx, id_keys=id_keys
            )
            prediction_cache.append({**id_dict, **pred_dict})

            # file splitting
            if (idx != 0) and (idx % num_records_per_file) == 0:
                self.write_once(
                    last_idx,
                    idx,
                    file_suffix,
                    num_samples,
                    num_records_per_file,
                    prediction_cache,
                    content_encoding,
                    "JSON",
                )
                last_idx = idx + 1
                prediction_cache = []

        # spit tail
        if len(prediction_cache) > 0:
            self.write_once(
                last_idx,
                idx,
                file_suffix,
                num_samples,
                num_records_per_file,
                prediction_cache,
                content_encoding,
                "JSON",
            )

    def write_predictions_to_csv_files(
        self,
        prediction_result_generator: list,
        num_records_per_file: int = default_num_lines_per_file,
        num_samples: int = 0,
        content_encoding: str = None,
    ) -> None:
        """Write predictions into a CSV file.

        Arguments:
            prediction_result_generator {list} -- A yield of prediction result.

        Keyword Arguments:
            num_records_per_file {int} -- The number of records in each output file. (default: {default_num_lines_per_file})
            num_samples {int} -- the number of samples to be included in the sample.json; this int number shall be smaller than num_records_per_file (default: {0})
            content_encoding {str} -- The encoding of the file content, that can be None or "GZIP". (default: {None})
        """

        prediction_cache = []
        last_idx, idx = 0, 0
        file_suffix = self.prediction_file_format["CSV"]
        id_keys = self.test_data_source.train_dev_conf.id_column_names

        for idx, entry in enumerate(prediction_result_generator, start=0):

            # transform and split raw prediction record
            id_dict, pred_dict = self.prediction_record_to_dict(
                record=entry, id_keys=id_keys, idx=idx
            )
            prediction_cache.append({**id_dict, **pred_dict})

            # file splitting
            if (idx != 0) and (idx % num_records_per_file) == 0:
                self.write_once(
                    last_idx,
                    idx,
                    file_suffix,
                    num_samples,
                    num_records_per_file,
                    prediction_cache,
                    content_encoding,
                    "CSV",
                )
                last_idx = idx + 1
                prediction_cache = []

        # spit tail
        if len(prediction_cache) > 0:
            self.write_once(
                last_idx,
                idx,
                file_suffix,
                num_samples,
                num_records_per_file,
                prediction_cache,
                content_encoding,
                "CSV",
            )

    def write_predictions_to_txt_files(
        self,
        prediction_result_generator: list,
        num_records_per_file: int = default_num_lines_per_file,
        num_samples: int = 0,
        content_encoding: str = None,
    ) -> None:
        """Output the prediction results in *.txt format.

        Arguments:
            prediction_result_generator {list} -- A yield of prediction result.

        Keyword Arguments:
            num_records_per_file {int} -- The number of records in each output file. (default: {default_num_lines_per_file})
            num_samples {int} -- the number of samples to be included in the sample.json; this int number shall be smaller than num_records_per_file (default: {0})
            content_encoding {str} -- The encoding of the file content, that can be None or "GZIP". (default: {None})
        """

        prediction_cache = []
        last_idx, idx = 0, 0
        file_suffix = self.prediction_file_format["TEXT"]
        for idx, entry in enumerate(prediction_result_generator, start=0):

            prediction_cache.append(entry)

            # file splitting
            if (idx != 0) and (idx % num_records_per_file) == 0:
                self.write_once(
                    last_idx,
                    idx,
                    file_suffix,
                    num_samples,
                    num_records_per_file,
                    prediction_cache,
                    content_encoding,
                    "TEXT",
                )
                last_idx = idx + 1
                prediction_cache = []

        # spit tail
        if len(prediction_cache) > 0:
            self.write_once(
                last_idx,
                idx,
                file_suffix,
                num_samples,
                num_records_per_file,
                prediction_cache,
                content_encoding,
                "TEXT",
            )
