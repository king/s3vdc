"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Tuple


def metric_simplified_silhouette(
    C: tf.Tensor, E: tf.Tensor, p: tf.Tensor, dist_type: str = "euclidean"
) -> Tuple[tf.Tensor, tf.Operation]:
    """Performs calculation of Silhouette Coefficient as a evaluation metric.
    
    Assume we obtained k clusters via certain unsupervised clustering algorithms.
    For each sample in evaluation/testing dataset, we can obtain the cluster it belongs (
    and the embedded space for some models). For the i-th sample, we calculate:
    - a(i) = mean(all distances from vector i to all other samples in the same cluster as i)
    - b(i) = min(mean distance to all samples in other clusters)
    Silhouette Coefficient = (b_i-a_i)/(max(a_i,b_i))

    To reduce the computation complexity, we adopt the following simplifications:
    - use the distance from a sample to cluster center to replace the mean distance to all
    points in a cluster.

    Arguments:
        C {tf.Tensor} -- The matrix (clusters * codes) denoting the centers of clusters.
        E {tf.Tensor} -- The embedding matrix (N * codes) for the samples.
        p {tf.Tensor} -- The column vector (of size N) containing the predicted cluster numbers.

    Keyword Arguments:
        dist_type {str} -- The type of distance metric. (default: {"euclidean"})

    Returns:
        Tuple[tf.Tensor, tf.Operation] -- A float metric containing the simplified Silhouette score.
    """

    num_centroids = tf.shape(C)[0]

    # calc distance matrix
    Ce = tf.expand_dims(C, 1)
    Ee = tf.expand_dims(E, 0)
    D = tf.sqrt(tf.reduce_sum(tf.squared_difference(Ce, Ee), -1))

    # construct binary matrix R
    _idx = tf.range(0, num_centroids)
    R = tf.cast(
        tf.equal(
            tf.reshape(p, (-1, tf.shape(p)[-1], 1)),
            tf.reshape(_idx, (-1, 1, tf.shape(_idx)[-1])),
        ),
        tf.float32,
    )
    R = tf.transpose(tf.squeeze(R))
    _R = tf.abs(R - 1.0)

    # approximate a's and b's
    a_s = tf.reduce_sum(D * R, 0)
    DmR = D * _R
    DmR.set_shape([None, None])

    b_s = tf.reduce_min(
        tf.boolean_mask(DmR, tf.greater(DmR, 0)), 0  # remove distance to itself
    )

    # calc Silhouette Coefficient
    # Formula: S_i=\frac{b_i-a_i}{max\{a_i,b_i\}}
    _result = (b_s - a_s) / tf.maximum(a_s, b_s)

    return tf.metrics.mean(_result)
