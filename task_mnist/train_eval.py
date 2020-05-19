"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


from lib.data_utils import DataUtils
from lib.traindev_config_parser import TrainDevConfigParser
from lib.tf_feature_columns import TfFeatureColumns
from lib.data_source import DataSource
from lib.trainer import Trainer
from lib.flag_parser import FlagParser
from lib.file_logger import FileLogger
import shutil
import tensorflow as tf
import os
import task_mnist


_this_root = os.path.join(os.path.dirname(task_mnist.__file__), "../")
shutil.copyfile(
    os.path.join(_this_root, "task_mnist/s3vdc.py"),
    os.path.join(_this_root, "lib/user_custom_model.py"),
)
from lib.resolve_model_type import resolve_model_type

model_dir = "model"
date_time_str = None
parsed_args = FlagParser.parse_args()

if parsed_args:
    date_time_str = parsed_args.datetime
else:
    raise RuntimeError(
        "missing datetime or pred-data string for training/evaluation job"
    )

if parsed_args.job_dir:
    model_dir = parsed_args.job_dir

FileLogger("TRAIN_EVAL", model_dir)

data_utils = DataUtils()
conf_parser = TrainDevConfigParser(
    data_utils.get_file_contents(
        os.path.join(os.path.dirname(task_mnist.__file__), "config.json")
    )
)

tf_feature_cols = TfFeatureColumns(conf_parser)

ds_train = DataSource(
    _tf_feature_cols=tf_feature_cols, _ds_class="train", _dsutil=data_utils
)
# train data_sources can not be empty
if ds_train.is_empty:
    raise RuntimeError("training dataset can not be empty!")

# evaluation dataset can be empty, yet a warning will be given
ds_eval = DataSource(
    _tf_feature_cols=tf_feature_cols, _ds_class="eval", _dsutil=data_utils
)
if ds_eval.is_empty:
    ds_eval = DataSource(
        _tf_feature_cols=tf_feature_cols,
        _ds_class="train",
        _dsutil=data_utils,
        force_test=True,
    )
    tf.logging.warn(
        "evaluation dataset is empty; will use training dataset for evaluation instead!"
    )

_handle = Trainer(
    model_fn=resolve_model_type(conf_parser.model_type),
    train_data_source=ds_train,
    eval_data_source=ds_eval,
    hyper_params=conf_parser.model_params,
    model_dir=model_dir,
    date_time_str=date_time_str,
)
_result = _handle.run()

tf.logging.info(_result)
