"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import numpy as np
from sklearn import mixture
from sklearn import cluster
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import logging as tfl
from tensorflow import feature_column as tfc
from tensorflow import estimator as tfe
from lib.custom_model import custom_model
from lib.resolve_activation_fn import resolve_activation_fn
from lib.resolve_optimizer import resolve_optimizer
from lib.resolve_simple_number import resolve_simple_number
from lib.resolve_simple_bool import resolve_simple_bool
from lib.cluster_separation import metric_cluster_separation
from lib.simplified_silhouette import metric_simplified_silhouette
from lib.calinski_harabaz import metric_calinski_harabaz
from lib.resolve_simple_number_array import resolve_simple_number_array
from typing import Tuple, Union


def get_cvae(
    hparams, _input: tf.Tensor, global_step: Union[tf.Tensor, int]
) -> Tuple[
    list,
    tf.Tensor,
    tf.layers.Dense,
    list,
    tf.layers.Dense,
    tf.layers.Dense,
    tf.Tensor,
    list,
    tf.Tensor,
]:
    """Obtain a convolution VAE. 
    Duplicate the Network definition in each task for task isolation.

    Arguments:
        hparams {[type]} -- Hyper parameters.
        _input {tf.Tensor} -- The input Tensor.
        global_step {Union[tf.Tensor, int]} -- Global step to use for the decay computation. Must not be negative.

    Returns:
        Tuple[list,tf.Tensor,tf.layers.Dense,list,tf.layers.Dense,tf.layers.Dense,tf.Tensor,list,tf.Tensor,] -- The constructed DCNN.
            - encoder_cnn {list} - The list of encoder CNN layers.
            - encoder_cnn_out {tf.Tensor} - The output of the encoder.
            - mu_layer {tf.layers.Dense} - The FC layer that generates mu.
            - code {list} - The Tensors (mu, sigma, and z) that constitute the code.
            - sigma_layer {tf.layers.Dense} - The FC layer that generates sigma.
            - decoder_dense {tf.layers.Dense} - The FC layer right after the bottleneck layer.
            - decoder_dense_out {tf.Tensor} - The output of the decoder_dense layer.
            - decoder_cnn {list} - The list of decoder CNN layers.
            - result {tf.Tensor} - The final decoded output.
    """

    # resolve filters
    filters = resolve_simple_number_array(hparams, "filters", int)  # type: List[int]
    n_filters = len(filters)
    if n_filters < 1:
        raise ValueError(
            "there should be at least one convolution layer defined in filters"
        )

    # resolve kernelSize
    kernel_size = resolve_simple_number_array(
        hparams, "kernelSize", int
    )  # type: List[int]
    n_kernel_size = len(kernel_size)
    if n_filters != n_kernel_size:
        raise ValueError(
            "expect {} kernelSizes in parameters, actual {} kernelSizes found".format(
                n_filters, n_kernel_size
            )
        )

    # resolve strides
    strides = resolve_simple_number_array(hparams, "strides", int)  # type: List[int]
    n_strides = len(strides)
    if n_strides != n_filters:
        raise ValueError(
            "expect {} strides in parameters, actual {} strides found".format(
                n_filters, n_strides
            )
        )
    # compute scale
    accum_scale = 1
    for entry in strides:
        accum_scale *= entry

    # resolve interp
    interp = resolve_simple_number_array(hparams, "interp", int)
    if interp is None:
        interp = resolve_simple_number_array(hparams, "frameShape", int)
    n_interp = len(interp)
    if n_interp != 2:
        raise ValueError(
            "expect 2 interp values in parameters, actual {} found".format(n_interp)
        )
    if interp[0] % accum_scale == 0 and interp[1] % accum_scale == 0:
        first_deconv_pad = "same"
    else:
        first_deconv_pad = "valid"

    # resolve numCode
    num_code = resolve_simple_number(hparams, "numCode")  # type: int

    """
    Construct the encoder convolution layers
    """
    result = _input
    encoder_cnn = []
    for i in range(n_filters):
        _layer = tf.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size[i],
            kernel_regularizer=tfk.regularizers.l2(l=0.01),
            bias_regularizer=tfk.regularizers.l2(l=0.01),
            strides=(strides[i], strides[i]),
            padding=first_deconv_pad if i == n_filters - 1 else "same",
            activation=resolve_activation_fn(hparams),
            trainable=True,
        )
        result = _layer.apply(result)
        encoder_cnn.append(_layer)

    encoder_cnn_out = result
    dim_before_flatten = tf.shape(encoder_cnn_out)
    last_encoder_layer_out = tf.layers.Flatten()(encoder_cnn_out)

    """
    bottleneck layer, i.e. embedding/codes are extracted from here
    """
    code = []
    mu_layer = tf.layers.Dense(
        units=num_code,
        trainable=True,
        kernel_regularizer=tfk.regularizers.l2(l=0.01),
        bias_regularizer=tfk.regularizers.l2(l=0.01),
        activation=None,
    )
    z_mu = mu_layer.apply(last_encoder_layer_out)
    code.append(z_mu)
    sigma_layer = tf.layers.Dense(
        units=num_code,
        trainable=True,
        kernel_regularizer=tfk.regularizers.l2(l=0.01),
        bias_regularizer=tfk.regularizers.l2(l=0.01),
        activation=None,
    )
    z_sigma = sigma_layer.apply(last_encoder_layer_out)
    code.append(z_sigma)
    eps = tf.random_normal(shape=tf.shape(z_sigma), mean=0, stddev=1, dtype=tf.float32)
    z = z_mu + tf.exp(z_sigma / 2) * eps
    code.append(z)
    result = tf.cond(
        global_step <= (hparams.gammaSteps + hparams.gmmSteps), lambda: z_mu, lambda: z
    )

    """
    Construct the decoder convolution layers
    """
    flatten_dim = (
        filters[-1] * int(interp[0] / accum_scale) * int(interp[1] / accum_scale)
    )
    # first dense layer
    decoder_dense = tf.layers.Dense(
        flatten_dim,
        kernel_regularizer=tfk.regularizers.l2(l=0.01),
        bias_regularizer=tfk.regularizers.l2(l=0.01),
        activation=resolve_activation_fn(hparams),
        trainable=True,
    )
    result = decoder_dense.apply(result)
    # need to reshape before input to CNNs
    result = tf.reshape(result, shape=dim_before_flatten)
    decoder_dense_out = result
    # deconv layers till the last
    decoder_cnn = []
    for i in range(n_filters - 2, -2, -1):
        _layer = tf.layers.Conv2DTranspose(
            filters=_input.shape[-1] if i < 0 else filters[i],
            kernel_size=kernel_size[i + 1],
            kernel_regularizer=tfk.regularizers.l2(l=0.01),
            bias_regularizer=tfk.regularizers.l2(l=0.01),
            strides=(strides[i + 1], strides[i + 1]),
            padding=first_deconv_pad if i == n_filters - 2 else "same",
            activation=resolve_activation_fn(hparams),
            trainable=True,
        )
        result = _layer.apply(result)
        decoder_cnn.append(_layer)

    return (
        encoder_cnn,
        encoder_cnn_out,
        mu_layer,
        code,
        sigma_layer,
        decoder_dense,
        decoder_dense_out,
        decoder_cnn,
        result,
    )


class S3VDCHook(tfe.SessionRunHook):
    """S3VDC Training flow control"""

    def __init__(
        self,
        vae: dict,
        vgmm: dict,
        hps: Union[tf.contrib.training.HParams, dict],
        utils: dict,
        is_chief: bool,
    ) -> None:
        """Initialize the S3VDCHook.

        Arguments:
            vae {dict} -- The tuple return by calling function get_cvae().
            vgmm {dict} -- The global GMM model weights/paramters.
            hps {Union[tf.contrib.training.HParams, dict]} -- Hyper parameters.
            utils {dict} -- The utility Tensors to be used by this class.
            is_chief {bool} -- Is chief node (only useful in distributed training).
        """

        self.vae = vae
        self.vgmm = vgmm
        self.hps = hps
        self.utils = utils
        self.is_chief = is_chief

        self.gamma_steps = resolve_simple_number(hps, "gammaSteps")
        self.gmm_steps = resolve_simple_number(hps, "gmmSteps")
        self.gamma_gmm_training_steps = self.gamma_steps + self.gmm_steps
        self.beta_steps = resolve_simple_number(hps, "betaSteps")
        self.static_steps = resolve_simple_number(hps, "staticSteps")
        self.svdc_gmm_finetune_steps = self.beta_steps + self.static_steps
        self.max_steps = resolve_simple_number(hps, "maxSteps")

        self.mu = tf.placeholder(dtype=tf.float32)
        self.sigma = tf.placeholder(dtype=tf.float32)

        # reversed anneal_factor can be used as an approximation of gmm update momentum
        self.mtm = tf.pow(1.0 - self.vgmm["anneal_factor"], 3)
        self.update_svdc_with_gmm_ops = [
            tf.assign(
                self.vgmm["sigma"],
                (1.0 - self.mtm) * self.vgmm["sigma"] + self.mtm * self.sigma,
            )
        ]
        self.update_svdc_with_kmeans_ops = [
            tf.assign(
                self.vgmm["mu"], (1.0 - self.mtm) * self.vgmm["mu"] + self.mtm * self.mu
            )
        ]

        self.gmm_data = None

    def skl_init(self, session: tf.Session, step: int) -> None:
        """Perform initialization of global GMM model.

        Arguments:
            session {tf.Session} -- Session from the context.
            step {int} -- Current training step.
        """

        is_final = True if step == 0 else False
        # gmm input data accumulation
        gmm_input_slice = session.run(self.vae["z_mu"])
        if self.gmm_data is None:
            self.gmm_data = gmm_input_slice
        else:
            self.gmm_data = np.concatenate([self.gmm_data, gmm_input_slice], axis=0)
        tfl.info(
            "GMM data accumulation done at local step {} with {} samples.".format(
                step, len(self.gmm_data)
            )
        )

        # gmm prediction
        if is_final:
            np.random.shuffle(self.gmm_data)
            kmeans = cluster.KMeans(n_clusters=self.hps.numCluster, random_state=0)
            kmeans.fit(self.gmm_data)
            session.run(
                self.update_svdc_with_kmeans_ops,
                feed_dict={self.mu: kmeans.cluster_centers_.T,},
            )
            skgmm = mixture.GaussianMixture(
                n_components=self.hps.numCluster,
                covariance_type="diag",
                max_iter=10000,
                means_init=kmeans.cluster_centers_,
                random_state=100,
            )
            skgmm.fit(self.gmm_data)
            session.run(
                self.update_svdc_with_gmm_ops,
                feed_dict={self.sigma: skgmm.covariances_.T},
            )
            self.gmm_data = None
            tfl.info(
                "GMM init done at local step {} with momentum factor {}".format(
                    step, session.run(self.mtm)
                )
            )

        session.run(self.utils["incr_gs_step"])

    def before_run(self, run_context: tfe.SessionRunContext) -> None:
        """The S3VDC flow control

        Arguments:
            run_context {tf.estimator.SessionRunContext} -- The run context.
        """

        session = run_context.session
        gs = tf.train.global_step(session, self.utils["global_step"])
        if gs == 0:
            tfl.info("S3VDC training begin at global step 1.")
            session.run(self.utils["incr_gs_step"])
            return
        if gs == self.max_steps - 1:
            tfl.info(
                "S3VDC training end at global step {} with latent annealing factor {}.".format(
                    gs, session.run(self.vgmm["anneal_factor"])
                )
            )
        # gs_offset: <=0 - vae/gmm training, >0 - svdc/gmm finetunes
        gs_offset = gs - self.gamma_gmm_training_steps
        # vae/gmm training
        if gs_offset <= 0:
            # vae training
            if gs_offset <= -1.0 * self.gmm_steps:
                if gs_offset == 1 - self.gamma_gmm_training_steps:
                    tfl.info("Gamma training begin at global step {}".format(gs))
                session.run(self.vgmm["gamma_training_op"])
            else:
                # gmm init steps
                self.skl_init(session, gs_offset)
        else:
            # svdc/gmm finetunes
            cyc_step = gs_offset % self.svdc_gmm_finetune_steps
            cyc_id = (gs_offset // self.svdc_gmm_finetune_steps) + 1
            # svdc finetune
            if cyc_step > 0 and cyc_step <= self.beta_steps:
                if cyc_step == 1:
                    tfl.info(
                        "PERIOD {}: Beta annealing begin at global step {} with latent annealing factor {}".format(
                            cyc_id, gs, session.run(self.vgmm["anneal_factor"])
                        )
                    )
                elif cyc_step == self.beta_steps:
                    tfl.info(
                        "PERIOD {}: Beta annealing end at global step {} with latent annealing factor {}".format(
                            cyc_id, gs, session.run(self.vgmm["anneal_factor"])
                        )
                    )
                session.run(self.vgmm["svdc_finetune_op"])
            else:
                # static training
                session.run(self.vgmm["svdc_static_train_op"])


@custom_model
def estimator_fn(
    only_features: dict,
    labels: dict,
    hparams: Union[tf.contrib.training.HParams, dict],
    mode: str,
    only_feature_columns: list,
    label_feature_columns: list,
    no_export_columns: list,
    config: tf.estimator.RunConfig,
) -> Tuple[dict, tf.Tensor, tf.Operation, dict, list, list]:
    """The implementation of a recommendation model.

    Arguments:
        only_features {dict} -- The dict of feature.
        labels {dict} -- The dict of label.
        hparams {Union[tf.contrib.training.HParams, dict]} -- The hyper-parameters of the model
        mode {str} -- The model during the training/evaluation process.
        only_feature_columns {list} -- The list of feature columns.
        label_feature_columns {list} -- The list of label columns.
        no_export_columns {list} -- The list of column names not to be included in the serving model.
        config {tf.estimator.RunConfig} -- training config

    Returns:
        Tuple[dict, tf.Tensor, tf.Operation, dict, list, list]
            - predictions {dict} - The prediction output Tensors.
            - loss {tf.Tensor} - The loss Tensor.
            - train_op {tf.Operation} - The training operation.
            - eval_metric_ops {dict} - Evaluation metrics.
            - training_hooks {list} - Custom training hooks.
            - eval_hooks {list} - Custom evaluation hooks.
    """

    utils = {}
    utils["global_step"] = tf.train.get_or_create_global_step()
    # input data blob
    input_feature = tfc.input_layer(only_features, only_feature_columns)
    utils["batch_size"] = tf.shape(input_feature)[0]
    frame_shape = resolve_simple_number_array(hparams, "frameShape")
    input_feature = tf.expand_dims(
        tf.reshape(
            input_feature, tf.convert_to_tensor([utils["batch_size"]] + frame_shape)
        ),
        axis=-1,
    )
    # interpolation
    interp = resolve_simple_number_array(hparams, "interp")
    if interp is not None and interp != frame_shape:
        input_feature = tf.image.resize_images(
            input_feature, tf.convert_to_tensor(interp)
        )
    input_feature_noise = input_feature
    if mode == tfe.ModeKeys.TRAIN:
        input_feature_noise += tf.random_normal(
            shape=tf.shape(input_feature), mean=0.0, stddev=1e-8, dtype=tf.float32
        )
    # runtime input dimension
    input_dim = tf.shape(input_feature)[1] * tf.shape(input_feature)[2]
    # the number of feature dimensions
    latent_dim = resolve_simple_number(hparams, "numCode")
    # the number of total expected clusters
    num_cluster = resolve_simple_number(hparams, "numCluster")

    # Variational AE
    with tf.variable_scope("VAE"):

        vae = dict(
            zip(
                [
                    "encoder_cnn",
                    "encoder_cnn_out",
                    "mu_layer",
                    "code",
                    "sigma_layer",
                    "decoder_dense",
                    "decoder_dense_out",
                    "decoder_cnn",
                    "output",
                ],
                get_cvae(hparams, input_feature_noise, utils["global_step"]),
            )
        )
        predictions = dict(zip(["z_mu", "z_sigma", "z"], vae["code"]))
        vae.update(predictions)

        # Vanilla VAE latent loss
        vae["latent_loss"] = (
            1 + vae["z_sigma"] - tf.square(vae["z_mu"]) - tf.exp(vae["z_sigma"])
        )
        vae["latent_loss"] = tf.reduce_sum(vae["latent_loss"], axis=-1)
        vae["latent_loss"] *= -0.5
        vae["latent_loss"] = tf.reduce_mean(vae["latent_loss"])

        # Tensorboard
        tf.summary.histogram(values=vae["z_mu"], name="z_mu")
        tf.summary.histogram(values=vae["z_sigma"], name="z_sigma")
        tf.summary.histogram(values=vae["z"], name="z")
        tf.summary.scalar("latent_loss", vae["latent_loss"])

    # GMM that works together with VAE
    with tf.device("/device:CPU:0"):
        with tf.variable_scope("VGMM"):

            log2pi = 1.8378770664093453

            vgmm = dict(
                zip(
                    ["pi_theta", "mu", "sigma"],
                    [
                        tf.get_variable(
                            name="pi_theta",
                            initializer=tf.truncated_normal(
                                shape=(num_cluster,),
                                mean=(1.0 / num_cluster),
                                stddev=(0.1 / num_cluster),
                                dtype=tf.float32,
                            ),
                            dtype=tf.float32,
                            trainable=True,
                        ),
                        tf.get_variable(
                            name="mu",
                            initializer=tf.truncated_normal(
                                shape=(latent_dim, num_cluster),
                                mean=0.0,
                                stddev=0.1,
                                dtype=tf.float32,
                            ),
                            dtype=tf.float32,
                            trainable=True,
                        ),
                        tf.get_variable(
                            name="sigma",
                            initializer=tf.truncated_normal(
                                shape=(latent_dim, num_cluster),
                                mean=1.0,
                                stddev=0.1,
                                dtype=tf.float32,
                            ),
                            dtype=tf.float32,
                            trainable=True,
                        ),
                    ],
                )
            )

            # obtain assignment prediction
            precisions = 1.0 / vgmm["sigma"]
            precisions_chol = 1.0 / tf.sqrt(vgmm["sigma"])
            _log_prob = tf.reduce_sum((tf.square(vgmm["mu"]) * precisions), axis=0)
            _log_prob -= 2.0 * tf.matmul(vae["z_mu"], vgmm["mu"] * precisions)
            _log_prob += tf.matmul(tf.square(vae["z_mu"]), precisions)
            log_det_chol = tf.reduce_sum(tf.log(precisions_chol), axis=0)
            log_prob = -0.5 * (latent_dim * log2pi + _log_prob) + log_det_chol
            log_weights = tf.log(vgmm["pi_theta"])

            _p_cz = -(log_prob + log_weights)

            _p_cz = tf.div(
                tf.subtract(_p_cz, tf.reduce_min(_p_cz)),
                tf.subtract(tf.reduce_max(_p_cz), tf.reduce_min(_p_cz)) + 1e-12,
            ) * (-20.0)

            vgmm["_p_cz"] = _p_cz
            predictions["gamma"] = vgmm["gamma"] = tf.nn.softmax(_p_cz + 1e-10, axis=-1)
            predictions["assignments"] = tf.argmax(
                predictions["gamma"], axis=-1, output_type=tf.int32
            )

            # S3VDC latent loss
            _mu = tf.tile(
                tf.expand_dims(vgmm["mu"], 0), multiples=[utils["batch_size"], 1, 1]
            )
            _sigma = tf.tile(
                tf.expand_dims(vgmm["sigma"], 0), multiples=[utils["batch_size"], 1, 1]
            )
            _z_mu = tf.tile(
                tf.expand_dims(vae["z_mu"], -1), multiples=[1, 1, num_cluster]
            )
            _z_sigma = tf.tile(
                tf.expand_dims(vae["z_sigma"], -1), multiples=[1, 1, num_cluster]
            )

            vgmm["gamma"] = tf.nn.softmax(_p_cz, axis=-1)  # predictions['gamma']
            _gamma = tf.expand_dims(vgmm["gamma"], 1)

            ###

            vgmm["latent_loss"] = -0.5 * tf.reduce_sum(1.0 + vae["z_sigma"], axis=1)
            vgmm["latent_loss"] -= tf.reduce_sum(
                vgmm["gamma"]
                * tf.log(
                    tf.tile(
                        tf.expand_dims(vgmm["pi_theta"], 0),
                        multiples=[utils["batch_size"], 1],
                    )
                ),
                axis=1,
            )
            vgmm["latent_loss"] += tf.reduce_sum(
                vgmm["gamma"] * tf.log(vgmm["gamma"]), axis=1
            )
            vgmm["latent_loss"] += 0.5 * tf.reduce_sum(
                _gamma
                * (
                    tf.log(_sigma)
                    + (tf.square(_z_mu - _mu) + tf.exp(_z_sigma)) / _sigma
                    + latent_dim * log2pi
                ),
                axis=[1, 2],
            )
            vgmm["latent_loss"] = tf.reduce_mean(vgmm["latent_loss"])

            # S3VDC reconstruction loss
            if resolve_simple_bool(hparams, "useSigmoidReconstructionLoss"):
                vgmm["reconstruction_loss"] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=input_feature, logits=vae["output"]
                    )
                )
            else:
                vgmm["reconstruction_loss"] = tf.losses.mean_squared_error(
                    labels=input_feature, predictions=vae["output"]
                )
                # this is an important fix
                vgmm["reconstruction_loss"] *= tf.cast(input_dim, dtype=tf.float32)

            vgmm["svdc_static_loss"] = vgmm["reconstruction_loss"] + vgmm["latent_loss"]

            # Tensorboard
            tf.summary.histogram(values=vgmm["gamma"], name="gamma")
            tf.summary.histogram(values=vgmm["pi_theta"], name="pi_theta")
            tf.summary.histogram(values=vgmm["mu"], name="mu")
            tf.summary.histogram(values=vgmm["sigma"], name="sigma")
            tf.summary.scalar("latent_loss", vgmm["latent_loss"])
            tf.summary.scalar("reconstruction_loss", vgmm["reconstruction_loss"])

    tf.summary.histogram(values=predictions["assignments"], name="assignments")

    # Init values to be returned
    eval_metric_ops, training_hooks = None, None
    dummy_train_op = tf.constant(0)

    # Train/Eval/Test Flows
    if mode != tfe.ModeKeys.PREDICT:

        utils["incr_gs_step"] = tf.assign_add(utils["global_step"], 1)
        cluster_centers_var = tf.transpose(vgmm["mu"])
        eval_metric_ops = {
            # Embedding
            "mean_absolute_error": tf.metrics.mean_absolute_error(
                labels=input_feature, predictions=vae["output"]
            ),
            "mean_squared_error": tf.metrics.mean_squared_error(
                labels=input_feature, predictions=vae["output"]
            ),
            "mean_cosine_distance": tf.metrics.mean_cosine_distance(
                labels=input_feature, predictions=vae["output"], dim=-1
            ),
            # Clustering
            "separation": metric_cluster_separation(cluster_centers_var),
            "EMBD_simple_silhouette": metric_simplified_silhouette(
                cluster_centers_var, vae["z_mu"], predictions["assignments"]
            ),
            "EMBD_calinski_harabaz": metric_calinski_harabaz(
                vae["z_mu"], predictions["assignments"], num_cluster
            ),
        }

        if mode == tfe.ModeKeys.TRAIN:

            optimizer, _ = resolve_optimizer(hparams, utils["global_step"])

            # Gamma phase: warm-up training
            vgmm["gamma_training_op"] = optimizer.minimize(
                vgmm["reconstruction_loss"] + hparams.gamma * vae["latent_loss"],
                global_step=utils["global_step"],
            )

            # Beta phase: annealing training
            vgmm["anneal_loss"] = vgmm["reconstruction_loss"]
            tbeta_ts = hparams.betaSteps + hparams.staticSteps
            post_training_step = tf.clip_by_value(
                utils["global_step"] - hparams.gammaSteps - hparams.gmmSteps,
                0,
                hparams.maxSteps,
            )
            m_var = tf.math.floordiv(post_training_step, tbeta_ts)
            vgmm["anneal_factor"] = (
                tf.clip_by_value(
                    post_training_step - m_var * tbeta_ts, 0, hparams.betaSteps
                )
                / hparams.betaSteps
            )
            vgmm["anneal_factor"] = (
                tf.cast(tf.pow(vgmm["anneal_factor"], 3), dtype=tf.float32)
                + hparams.gamma
            )
            tf.summary.scalar("anneal_factor", vgmm["anneal_factor"])
            vgmm["anneal_loss"] += vgmm["anneal_factor"] * vgmm["latent_loss"]
            tf.summary.scalar("anneal_loss", vgmm["anneal_loss"])
            vgmm["svdc_finetune_op"] = optimizer.minimize(
                vgmm["anneal_loss"], global_step=utils["global_step"]
            )

            # Static phase
            vgmm["svdc_static_train_op"] = optimizer.minimize(
                vgmm["svdc_static_loss"], global_step=utils["global_step"]
            )

            training_hooks = [S3VDCHook(vae, vgmm, hparams, utils, config.is_chief)]

    return (
        predictions,
        vgmm["svdc_static_loss"],
        dummy_train_op,
        eval_metric_ops,
        training_hooks,
        None,
    )
