"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import os
import json
import datetime
import tensorflow as tf
from lib.run_config import default_run_config
from lib.data_source import DataSource
from lib.hyper_parameters import join_hparams, default_general_hparams
from collections import namedtuple
from typing import Callable, Union, Tuple

# from tensorflow.python import debug as tf_debug

TrainDevResults = namedtuple("TrainDevResult", ["estimator", "metrics"])


class Trainer:
    """
    The implementation of a Trainer.
    """

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
        train_data_source: DataSource,
        eval_data_source: DataSource,
        hyper_params: Union[tf.contrib.training.HParams, dict] = None,
        hooks: list = None,
        model_dir: str = "model",
        date_time_str: str = None,
        export_models_as_text: bool = False,
    ) -> None:
        """Initialize a Trainer

        Arguments:
            model_fn {Callable[[dict,dict,tf.contrib.training.HParams,str,list,list,list,tf.estimator.RunConfig,],Tuple[dict, tf.Tensor, tf.Operation, dict, list, list],]} -- A model function.
            train_data_source {DataSource} -- The DataSource object of training dataset.
            eval_data_source {DataSource} -- The DataSource object of evaluation dataset.

        Keyword Arguments:
            hyper_params {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters. (default: {None})
            hooks {list} -- A list of custom hooks. (default: {None})
            model_dir {str} -- The model directory. (default: {"model"})
            date_time_str {str} -- A date and time string in the format of "yyyymmdd_hhmmss" (default: {None})
            export_models_as_text {bool} -- Export readable model if True. (default: {False})
        """

        # hyper_params
        if hyper_params is None:
            hyper_params = {}
        hyper_params = tf.contrib.training.HParams(**hyper_params)
        self.hyper_params = join_hparams(
            default_general_hparams(), hyper_params
        )  # type: tf.contrib.training.HParams

        # run_config
        # TODO: make run_config configurable
        self.model_dir = model_dir
        self.run_config = default_run_config(
            model_dir=model_dir,
            save_summary_steps=self.hyper_params.get(
                "summaryFrequency", default=100
            ),  # unit: steps
            save_checkpoints_mins=self.hyper_params.get(
                "checkpointFrequency", default=5
            ),  # unit: minutes
            keep_checkpoint_max=self.hyper_params.get("keepMaxCheckpoint", default=5),
        )

        # date and time string
        self.date_time_str = date_time_str

        # default eval metrics output path
        self.eval_metric_output_path = ""
        if date_time_str is None:
            tf.logging.warning(
                "missing date and time info as input params for training job"
            )
            self.eval_metric_output_path = os.path.join(
                model_dir,
                "eval_metric_{}.json".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
            )
        else:
            self.eval_metric_output_path = os.path.join(
                model_dir, "eval_metric_{}.json".format(date_time_str)
            )

        # TODO: extend the functions via different hooks
        self.hooks = hooks

        # set train and eval data sources
        self.train_data_source = train_data_source
        self.eval_data_source = eval_data_source

        # model_fn
        self.model_fn = model_fn

        # feature columns
        self.only_feature_columns = (
            train_data_source.tf_feature_cols.only_feature_columns
        )
        # self.id_feature_columns = train_data_source.tf_feature_cols.id_feature_columns
        # self.label_feature_columns = train_data_source.tf_feature_cols.label_feature_columns

        # trainspec
        self.train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_data_source.input_fn(
                batch_size=self.hyper_params.batchSize
            ),
            max_steps=self.hyper_params.maxSteps,
        )

        # Serving input function
        feature_spec = tf.feature_column.make_parse_example_spec(
            # self.train_data_source.tf_feature_cols.id_feature_columns
            # + self.train_data_source.tf_feature_cols.only_feature_columns
            self.train_data_source.tf_feature_cols.export_feature_columns
        )
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec
        )
        self.export_models_as_text = export_models_as_text
        best_exporter = tf.estimator.BestExporter(
            name="best_exporter",
            serving_input_receiver_fn=serving_input_fn,
            as_text=self.export_models_as_text,
        )
        final_exporter = tf.estimator.FinalExporter(
            name="final_exporter",
            serving_input_receiver_fn=serving_input_fn,
            as_text=self.export_models_as_text,
        )

        # estimator
        self.estimator = self.model_fn(
            self.only_feature_columns,
            self.run_config,
            self.hyper_params,
            train_data_source.tf_feature_cols,
        )

        # evalspec
        throttle_secs = self.hyper_params.get(
            "minEvalFrequency", default=10
        )  # unit: minutes
        self.eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_data_source.input_fn,
            steps=None,
            exporters=[final_exporter, best_exporter],
            start_delay_secs=0,
            throttle_secs=throttle_secs * 60
            # hooks=[tf_debug.DumpingDebugHook("/tmp/tfdb_dump1")]
        )

    def run(self) -> TrainDevResults:
        """Run an experiment using the model defined via model_fn and hparams using train_dataset for training,
        eval_dataset for evaluation during training, and test_dataset for final model evaluation after training
        is finished.

        Returns:
            TrainDevResults -- An ExperimentResult instance.
        """

        eval_metrics = tf.estimator.train_and_evaluate(
            self.estimator, self.train_spec, self.eval_spec
        )

        # Temp fix due to https://github.com/tensorflow/tensorflow/issues/22417
        if eval_metrics is None:
            eval_metrics = self.estimator.evaluate(
                input_fn=self.eval_data_source.input_fn
            )
            traindev_results_metrics = eval_metrics
        else:
            traindev_results_metrics = eval_metrics[0]

        traindev_results = TrainDevResults(
            self.train_data_source.train_dev_conf.model_type, eval_metrics
        )

        if (
            traindev_results_metrics is not None
        ):  # continue training on a finished job will return None
            try:
                traindev_results_metrics_str = json.dumps(traindev_results_metrics)
            except TypeError:
                for metric_key in traindev_results_metrics:
                    traindev_results_metrics[metric_key] = traindev_results_metrics[
                        metric_key
                    ].item()
                traindev_results_metrics_str = json.dumps(traindev_results_metrics)

            # persist the test_metrics in a file either locally or in GCS
            self.eval_data_source.data_utils.write_to_file(
                traindev_results_metrics_str, self.eval_metric_output_path
            )
            tf.logging.info(
                "TrainDevResults.metrics are persisted in {}".format(
                    self.eval_metric_output_path
                )
            )

        return traindev_results
