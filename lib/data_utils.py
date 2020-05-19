"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import os
import tensorflow as tf
import gzip
import io


class DataUtils:
    """
    Definition of data utility functions
    """

    @staticmethod
    def gzip_str(_str: str) -> bytes:
        """Encode string to compressed bytes

        Arguments:
            _str {str} -- string to be compressed

        Returns:
            bytes -- compressed bytes
        """
        out = io.BytesIO()

        with gzip.GzipFile(fileobj=out, mode="w") as fo:
            fo.write(_str.encode())

        bytes_obj = out.getvalue()
        return bytes_obj

    def write_to_file(
        self, str_content: str, file_path: str, content_encoding: str = None
    ) -> None:
        """Write string to a (compressed) file

        Arguments:
            str_content {str} -- string to be written
            file_path {str} -- the path of the filed to be written to

        Keyword Arguments:
            content_encoding {str} -- perform compression upon value "gzip" (default: {None})
        """

        tf.logging.info("Writing file: {}".format(file_path))

        if content_encoding not in {None, "gzip"}:
            raise ValueError(
                "content_coding {} not supported in write_to_file".format(
                    str(content_encoding)
                )
            )

        if content_encoding is not None:
            file_path = "{}.{}".format(file_path, content_encoding)
            if content_encoding == "gzip":
                str_content = self.gzip_str(str_content)

        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if content_encoding == "gzip":
            with gzip.open(file_path, "wb") as file:
                file.write(str_content)
        else:
            with open(file_path, "w") as file:
                file.write(str_content)

    def get_file_contents(self, _path: str) -> str:
        """Return the content in a file located locally

        Arguments:
            _path {str} -- relative file path

        Returns:
            str -- the content of the file
        """

        with open(_path, "r") as _file:
            result = _file.read()
        return result
