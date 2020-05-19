"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Tuple


def metric_calinski_harabaz(
    X: tf.Tensor, L: tf.Tensor, num_clusters: int
) -> Tuple[tf.Tensor, tf.Operation]:
    """The resulting Calinski-Harabaz score of sample clusters.
    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion (i.e. Covariance).
    http://www.tandfonline.com/doi/abs/10.1080/03610927408827101

    Formula of CH score for the k-th cluster:
    CH(k) = \frac{trB(k)}{trW(k)} \times \frac{k-1}{n-k},
    where n is the total number of clusters; trB(k) is the trace of the extra-cluster
    covariance matrix (i.e. the sum of feature variance); and trW(k) denotes the trace
    of intra-cluster covariance matrix.

    Largely, the higher the score is, the better the cluster result is.

    Arguments:
        X {tf.Tensor} -- 2D Tensor of features
        L {tf.Tensor} -- 1D Tensor of predicted labels for samples
        num_clusters {int} -- the total number of clusters

    Returns:
        Tuple[tf.Tensor, tf.Operation] -- A float metric containing the Calinski-Harabaz score
    """

    num_samples = tf.shape(X)[0]

    extra_dispersion, intra_dispersion = 0.0, 0.0

    mean = tf.reduce_mean(X, axis=0)

    unique_L, _ = tf.unique(L)
    n_clusters_in_batch_k = tf.shape(unique_L)[0]

    for k in range(num_clusters):

        _mask = tf.equal(L, k)
        X_k = tf.boolean_mask(X, _mask)
        num_sample_k = tf.shape(X_k)[0]

        mean_k = tf.cond(
            tf.equal(num_sample_k, 0),
            lambda: tf.zeros(tf.shape(mean), dtype=tf.float32),
            lambda: tf.reduce_mean(X_k, axis=0),
        )

        extra_dispersion += tf.cast(num_sample_k, tf.float32) * tf.reduce_sum(
            tf.square((mean_k - mean))
        )
        intra_dispersion += tf.reduce_sum(tf.square(X_k - mean_k))

    (extra_disp_ops, extra_disp_mean) = tf.metrics.mean(extra_dispersion)
    (intra_disp_ops, intra_disp_mean) = tf.metrics.mean(intra_dispersion)

    nominator = extra_disp_mean * tf.cast(
        (num_samples - n_clusters_in_batch_k), tf.float32
    )
    denominator = intra_disp_mean * tf.cast((n_clusters_in_batch_k - 1), tf.float32)

    _result = tf.cond(
        tf.equal(denominator, 0.0), lambda: 1.0, lambda: nominator / denominator
    )

    return tf.metrics.mean(_result)
