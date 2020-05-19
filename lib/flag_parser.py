"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import argparse

FlagParser = argparse.ArgumentParser()

FlagParser.add_argument("--job-dir", help="the job dir")
FlagParser.add_argument(
    "--datetime", help="the date and time when the job is submitted"
)
FlagParser.add_argument(
    "--pred-data", help="specify which dataset to use when carrying out prediction job"
)
FlagParser.add_argument(
    "--file-format", help="specify the format of prediction outputs"
)
