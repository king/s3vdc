"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import tensorflow as tf
from typing import Tuple


def metric_cluster_separation(
    A: tf.Tensor, num_clusters: int = None
) -> Tuple[tf.Tensor, tf.Operation]:
    """Calculate the average distance between wither 2 cluster centers.
    Formula: (2/(k^2-k))\sum_{i=1}^k\sum_{j=i+1}^k||a_i-a_j||_2,
    where a_i represents the i-th row vector

    Implementation details:
    A key step of this calculation is to obtain the pairwise square distances of A,
    denoted as D. To turn it into an matrix op., we have:
    D[i,j] = (a[i]-a[j])(a[i]-a[j])'
    => D[i,j] = r[i] - 2 a[i]a[j]' + r[j],
    where r[i] is the squared norm of the i-th row of the original matrix;
    because of broadcasting, we can treat r as a column vector and hence D is
    D = r - 2 A A' + r'

    Arguments:
        A {tf.Tensor} -- The tensor of row Vectors of cluster centers

    Keyword Arguments:
        num_clusters {int} -- The number of clusters (default: {None})

    Returns:
        Tuple[tf.Tensor, tf.Operation] -- A float metric containing the Cluster Separation score
    """

    r = tf.reduce_sum(A * A, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

    if num_clusters is None:
        num_clusters = tf.to_float(tf.shape(A)[0])

    _result = tf.reduce_sum(tf.sqrt(tf.nn.relu(D))) / (num_clusters ** 2 - num_clusters)

    return tf.metrics.mean(_result)
