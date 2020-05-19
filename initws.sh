#!/bin/bash
#
# Copyright (C) king.com Ltd 2019
# https://github.com/king/s3vdc
# License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
#
# Initializing the workspace


if [[ $PWD != */s3vdc ]] ; then
    # Exit if not in S3VDC folder
    echo "Please run within S3VDC folder. Do nothing and exit!"
    exit 1
fi

echo "[-] Downloading preprocessed datasets and pretrained models."
wget https://storage.googleapis.com/s3vdc-limited-sharing/benchmark_tfdata_model.tar.gz
echo "Done."

echo "[-] Extracting files."
tar -xvf benchmark_tfdata_model.tar.gz
echo "Done."

echo "[-] Clearing temp files."
rm benchmark_tfdata_model.tar.gz
echo "Done."

echo "[-] Changing file access permissions."
chmod a+x ./*.sh
echo "Done."
