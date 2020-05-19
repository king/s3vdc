"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import logging
import os
import tensorflow as tf


class FileLogger:
    """
    FileLogger: manages the logs of training and testing jobs
    """

    def __init__(self, job_type: str, model_dir: str) -> None:
        """Initialize the FileLogger

        Arguments:
            job_type {str} -- job type
            model_dir {str} -- model_dir
        """
        self.job_type = job_type
        self.model_dir = model_dir
        tf.logging.set_verbosity(tf.logging.INFO)
        self.resolve_local_log_to_file()

    def resolve_local_log_to_file(self) -> None:
        """Set the logging according to job type
        """
        _log = logging.getLogger("tensorflow")
        _log.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        fh = logging.FileHandler(
            os.path.join(self.model_dir, "{}.log".format(self.job_type.lower()))
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        _log.addHandler(fh)
