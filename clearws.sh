#!/bin/bash
#
# Copyright (C) king.com Ltd 2019
# https://github.com/king/s3vdc
# License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
#
# Clearing the workspace


if [[ $PWD != */s3vdc ]] ; then
    # Exit if not in S3VDC folder
    echo "Please run within S3VDC folder. Do nothing and exit!"
    exit 1
fi

echo "[-] Remove user models, logs, and predictions."

rm -rf ./task_mnist/model
rm -rf ./task_inertial_har/model
rm -rf ./task_fashion/model

rm -rf ./models/mnist/*.log
rm -rf ./models/mnist/prediction_*_*

rm -rf ./models/inertial_har/*.log
rm -rf ./models/inertial_har/prediction_*_*

rm -rf ./models/fashion/*.log
rm -rf ./models/fashion/prediction_*_*

echo "Done."

if [ "$1" = "all" ] ; then
    echo "[-] Remove pretrained models and preprocessed datasets."
    rm -rf ./models
    rm -rf ./datasets
    echo "Done."
fi