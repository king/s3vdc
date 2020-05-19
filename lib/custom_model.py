"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


from functools import wraps
import tensorflow as tf
from lib.tf_feature_columns import TfFeatureColumns
from lib.data_source import DataSource
from typing import Callable
from typing import Tuple


def custom_model(
    user_model_fn: Callable[
        [dict, dict, tf.contrib.training.HParams, str],
        Tuple[dict, tf.Tensor, tf.Operation, dict],
    ]
) -> Callable[
    [list, tf.estimator.RunConfig, tf.contrib.training.HParams, TfFeatureColumns],
    tf.estimator.Estimator,
]:
    """Wraps a user defined model function to fit the framework by returning a function that
    returns a tf.estimator.Estimator based on the user defined model.

    Arguments:
        user_model_fn {Callable[[dict, tf.Tensor, tf.contrib.training.HParams, str],Tuple[dict, tf.Tensor, tf.Operation, dict],]}
            The model function to wrap. Model function should be of signature:
            Arguments:
                - features {dict} - A dict of features with feature names (str) as keys.
                - labels {dict} - A dict of labels with feature names (str) as keys. 
                - hparams {tf.contrib.learn.HParams} - hyper parameters.
                - mode {str}: one of tf.estimator.ModeKeys.(TRAIN|EVAL|PREDICT) to specify what the model should be built for.
            Returns:
                - predictions {list} - List of Tensors containing the predictions.
                - loss {tf.Tensor} - loss tensor.
                - train_op {tf.Operation} - training op.
                - eval_metric_ops {dict} - A dict of metric ops.

    Returns:
        Callable[[list, tf.estimator.RunConfig, tf.contrib.training.HParams, TfFeatureColumns],tf.estimator.Estimator,] 
            A function returning a tf.estimator.Estimator
    """

    @wraps(user_model_fn)
    def estimator_fn(
        only_feature_columns: list,
        run_config: tf.estimator.RunConfig,
        hparams: tf.contrib.training.HParams,
        tf_feature_col_obj: TfFeatureColumns,
    ) -> tf.estimator.Estimator:
        """[summary]

        Arguments:
            only_feature_columns {list} -- A list of feature columns
            run_config {tf.estimator.RunConfig} -- run config
            hparams {tf.contrib.training.HParams} -- hyper parameters
            tf_feature_col_obj {TfFeatureColumns} -- an instance of TfFeatureColumns

        Returns:
            tf.estimator.Estimator -- the built model estimator
        """

        if type(tf_feature_col_obj) is not TfFeatureColumns:
            raise TypeError(
                "tf_feature_col_obj should be of type TfFeatureColumns instead of {}".format(
                    str(type(tf_feature_col_obj))
                )
            )

        def model_fn(
            features: dict,
            labels: dict,
            mode: str,
            params: tf.contrib.training.HParams,
            config: tf.estimator.RunConfig,
        ) -> tf.estimator.EstimatorSpec:
            """Model Function

            Arguments:
                features {dict} -- A dict of features with feature names (str) as keys.
                labels {dict} -- A dict of labels with feature names (str) as keys.
                mode {str} -- one of tf.estimator.ModeKeys.(TRAIN|EVAL|PREDICT).
                params {tf.contrib.training.HParams} -- hyper parameters.
                config {tf.estimator.RunConfig} -- run config

            Returns:
                tf.estimator.EstimatorSpec -- Estimator specification
            """

            tf.logging.info("model_fn is called with mode={}".format(str(mode)))

            # hold Ids
            id_holder = {}
            id_feature_columns = tf_feature_col_obj.id_feature_columns
            if id_feature_columns is not None:
                with tf.variable_scope("ID_holder"):
                    for id_fc in id_feature_columns:
                        _id_fc_key = TfFeatureColumns.feature_column_key(id_fc)
                        id_holder[_id_fc_key] = DataSource.try_sparse2dense_tensor(
                            features[_id_fc_key]
                        )
                        del features[_id_fc_key]

            # Call the actual model function!
            (
                predictions,
                loss,
                train_op,
                eval_metric_ops,
                training_hooks,
                evaluation_hooks,
            ) = user_model_fn(
                only_features=features,
                labels=labels,
                hparams=params,
                mode=mode,
                only_feature_columns=only_feature_columns,
                label_feature_columns=tf_feature_col_obj.label_feature_columns,
                no_export_columns=tf_feature_col_obj.no_export_feature_columns,
                config=config,
            )

            # wrap exports
            exports = {"predictions": tf.estimator.export.PredictOutput(predictions)}

            # append Ids to prediction result
            for key in id_holder.keys():
                predictions[key] = id_holder[key]

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops,
                export_outputs=exports,
                training_hooks=training_hooks,
                evaluation_hooks=evaluation_hooks,
            )

        return tf.estimator.Estimator(
            model_fn=model_fn, config=run_config, params=hparams
        )

    return estimator_fn
