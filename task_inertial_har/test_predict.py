"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


from lib.data_utils import DataUtils
from lib.traindev_config_parser import TrainDevConfigParser
from lib.tf_feature_columns import TfFeatureColumns
from lib.data_source import DataSource
from lib.tester import Tester
from lib.resolve_model_type import resolve_model_type
from lib.flag_parser import FlagParser
from lib.file_logger import FileLogger
import shutil
import tensorflow as tf
import os
import task_inertial_har


_this_root = os.path.join(os.path.dirname(task_inertial_har.__file__), "../")
shutil.copyfile(
    os.path.join(_this_root, "task_inertial_har/s3vdc.py"),
    os.path.join(_this_root, "lib/user_custom_model.py"),
)
from lib.resolve_model_type import resolve_model_type

model_dir = "model"
date_time_str, pred_data = None, None
parsed_args = FlagParser.parse_args()
file_format = "JSON"

if parsed_args:
    date_time_str = parsed_args.datetime
    pred_data = parsed_args.pred_data
    file_format = parsed_args.file_format
else:
    raise RuntimeError("missing datetime or pred-data parameter for prediction job")

if parsed_args.job_dir:
    model_dir = parsed_args.job_dir

FileLogger("TEST_PREDICT", model_dir)

data_utils = DataUtils()
conf_parser = TrainDevConfigParser(
    data_utils.get_file_contents(
        os.path.join(os.path.dirname(task_inertial_har.__file__), "config.json")
    )
)

tf_feature_cols = TfFeatureColumns(conf_parser)

ds_test = DataSource(
    _tf_feature_cols=tf_feature_cols,
    _ds_class=pred_data,
    _dsutil=data_utils,
    force_test=True,
)
if ds_test.is_empty:
    ds_test = DataSource(
        _tf_feature_cols=tf_feature_cols,
        _ds_class="train",
        _dsutil=data_utils,
        force_test=True,
    )
    tf.logging.warn(
        "prediction dataset is empty; will use training dataset for prediction instead!"
    )

_handle = Tester(
    model_fn=resolve_model_type(conf_parser.model_type),
    model_dir=model_dir,
    test_data_source=ds_test,
    hyper_params=conf_parser.model_params,
    date_time_str=date_time_str,
)
_result = _handle.run_predictions(
    file_format=file_format, num_samples=10, content_encoding="gzip"
)

tf.logging.info(_result)
