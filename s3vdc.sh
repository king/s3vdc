#!/bin/bash
#
# Copyright (C) king.com Ltd 2019
# https://github.com/king/s3vdc
# License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
#
# Train, predict tasks on datasets MNIST, Inertial HAR, and FASHION


if [[ $PWD != */s3vdc ]] ; then
    # Exit if not in S3VDC folder
    echo "Please run within S3VDC folder. Do nothing and exit!"
    exit 1
fi

display_help() {
    echo
    echo "Usage: $0 {train|pred|rep} {mnist|inertial_har|fashion} [{json|csv|text}]" >&2
    echo
    exit 1
}

# check the first param: task
case "$1" in
  train)
    echo "Task: training/finetuning an S3VDC model." >&2
    ;;
  pred)
    echo "Task: prediction with user S3VDC model." >&2
    ;;
  rep)
    echo "Task: prediction with pretrained S3VDC model." >&2
    ;;
  *)
    echo "Invalid task \"$1\" !" >&2
    display_help
    exit 1
    ;;
esac
task=$1

# check the second param: dataset
case "$2" in
  mnist)
    echo "Dataset: MNIST." >&2
    ;;
  inertial_har)
    echo "Dataset: Inertial HAR." >&2
    ;;
  fashion)
    echo "Dataset: Fashion." >&2
    ;;
  *)
    echo "Invalid dataset \"$2\" !" >&2
    display_help
    exit 1
    ;;
esac
dataset=$2
task_name="task_$2"

# check the second param: dataset
file_format=$3
case "$3" in
  json)
    ;;
  csv)
    ;;
  text)
    ;;
  "")
    file_format="json" # default format
    ;;
  *)
    echo "Invalid file format \"$3\" !" >&2
    display_help
    exit 1
    ;;
esac
if [[ "$1" != "train" ]] ; then
    echo "File format of prediction output: $file_format." >&2
fi

# check availability of preprocessed datasets
if [ ! -d "./datasets/$2" ] 
then
    echo "Dataset $2 not found in workspace path ./datasets/$2." 
    echo "Please perform steps:"
    echo "(1) Run \"./clearws.sh all\" to reset workspace." 
    echo "(2) Run \"./initws.sh\" to initialize workspace." 
    exit 2 
fi

# check availability of the model trained by user for pred task
if [[ "$1" = "pred" && ! -d "./$task_name/model" ]]
then 
    echo "No user-trained S3VDC model found for prediction task on $2 dataset." 
    echo "Please train an S3VDC model first:"
    echo "Run \"./s3vdc.sh train $2\" to start training."
    exit 3
fi

# check availability of the pretrained model for rep task
if [[ "$1" = "rep" && ! -d "./models/$2" ]]
then 
    echo "No pretrained S3VDC model found for prediction task on $2 dataset." 
    echo "Please perform steps:"
    echo "(1) Run \"./clearws.sh all\" to reset workspace." 
    echo "(2) Run \"./initws.sh\" to initialize workspace."
    exit 4
fi

# build command
task_type="train_eval"
if [[ "$1" != "train" ]] ; then
    task_type="test_predict"
fi
job_dir="$task_name/model"
if [[ "$1" = "rep" ]] ; then
    job_dir="models/$2"
fi
pred_data_op="--pred-data test"
file_format_op="--file-format $file_format"
if [[ "$1" = "train" ]] ; then
    pred_data_op=""
    file_format_op=""
fi
full_cmd="python -m $task_name.$task_type --job-dir $job_dir $pred_data_op $file_format_op"
echo $full_cmd

# start the task
eval $full_cmd
