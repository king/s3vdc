"""
Copyright (C) king.com Ltd 2019
https://github.com/king/s3vdc
License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
"""


import json
import tensorflow as tf
from typing import Union


class TrainDevConfigParser:
    """
    Utilities of parsing the TrainDev configuration
    """

    _feature_dtypes = {"FLOAT": tf.float32, "INTEGER": tf.int64, "STRING": tf.string}

    def __init__(self, _conf: str) -> None:
        """Initialize a TrainDevConfigParser

        Arguments:
            _conf {str} -- The configs.
        """
        self._raw_conf = json.loads(_conf)
        self._model_type = self._raw_conf["modelType"]
        self._model_params = self._raw_conf["modelParam"][self._model_type]

        self._id_col_names = []
        # idCols is optional
        if "idCols" in self._raw_conf:
            self._id_col_names = list(self._raw_conf["idCols"])
            if len(self._id_col_names) > 1:
                self._id_col_names = sorted(self._id_col_names)

        # labelCols is optional
        self._label_col_names = []
        if "labelCols" in self._raw_conf:
            self._label_col_names = list(self._raw_conf["labelCols"])
            if len(self._label_col_names) > 1:
                self._label_col_names = sorted(self._label_col_names)

        # exceptCols is optional
        self._except_col_names = []
        if "exceptCols" in self._raw_conf:
            self._except_col_names = list(self._raw_conf["exceptCols"])
            if len(self._except_col_names) > 1:
                self._except_col_names = sorted(self._except_col_names)

        # exceptCols is optional
        self._no_export_col_names = []
        if "noExportCols" in self._raw_conf:
            self._no_export_col_names = list(self._raw_conf["noExportCols"])
            if len(self._no_export_col_names) > 1:
                self._no_export_col_names = sorted(self._no_export_col_names)

        self._data_source = self._raw_conf["dataSource"]

        # Transform object array to dict
        self._fcc_dict = {}
        for _fcc in self._raw_conf["featureColumns"]:
            _name = _fcc["name"]
            del _fcc["name"]
            self._fcc_dict[_name] = _fcc

        self._all_col_names = list(self._fcc_dict.keys())
        if len(self._all_col_names) > 1:
            self._all_col_names = sorted(self._all_col_names)

    @property
    def column_names(self) -> list:
        """Obtain all the column names, including imputation columns.

        Returns:
            list -- A sorted list of all feature column names.
        """
        return self._all_col_names

    @property
    def label_column_names(self) -> list:
        """Obtain all names for label columns.

        Returns:
            list -- A sorted list of all label feature column names.
        """
        return self._label_col_names

    @property
    def id_column_names(self) -> list:
        """Obtain all names for ID columns.

        Returns:
            list -- A sorted list of all ID feature column names.
        """
        return self._id_col_names

    @property
    def except_column_names(self) -> list:
        """Obtain all column names that shall be excluded from formal modelling.

        Returns:
            list -- A sorted list of all feature column names to be excluded.
        """
        return self._except_col_names

    @property
    def no_export_column_names(self) -> list:
        """Obtain the column names that shall not be included in the exported model.
        Note that these columns will not be directly accessible in mode=infer

        Returns:
            list -- A sorted list of all feature column names to be excluded from the exported model.
        """
        return self._no_export_col_names

    @property
    def feature_column_names(self) -> list:
        """Obtain all names of feature columns (i.e. excluding label and id columns)

        Returns:
            list -- A sorted list of feature columns.
        """
        only_feature_cols = list(
            set(self.column_names)
            - set(self.label_column_names)
            - set(self.id_column_names)
            - set(self.except_column_names)
        )
        if len(only_feature_cols) > 1:
            only_feature_cols = sorted(only_feature_cols)
        return only_feature_cols

    @property
    def model_type(self) -> str:
        """Obtain the type of the model to be trained

        Returns:
            [str] -- The name of the model type.
        """
        return self._model_type

    @property
    def model_params(self) -> dict:
        """Obtain the hyper-parameters for the model_type

        Returns:
            dict -- A dict of model hyper-parameters.
        """
        return self._model_params

    def set_model_params(self, model_params: dict) -> dict:
        """Set the model_param with the provided values.

        Arguments:
            model_params {dict} -- A dict of model hyper-parameters.

        Returns:
            dict -- The new model hyper-parameters.
        """
        self._model_params = model_params
        return model_params

    @property
    def data_source(self) -> dict:
        """Return the raw configuration of dataSource

        Returns:
            dict -- A dict of dataset configurations.
        """
        return self._data_source

    def feature_column_config(self, _name: str) -> dict:
        """Obtain the entire raw configuration for column _name.

        Arguments:
            _name {str} -- The feature name to search for.

        Returns:
            dict -- The found feature configuration (return None when not found).
        """
        _fcc = self._fcc_dict[_name]
        if _fcc is None:
            raise KeyError("{} is not a valid feature column".format(_name))
        return _fcc

    def feature_column_config_attribute(
        self, _name: str, _attribute_name: str
    ) -> Union[str, dict, list]:
        """Return the value of attribute _attribute_name from feature column config _name.

        Arguments:
            _name {str} -- The name of the target column
            _attribute_name {str} -- The attribute name in _feature_column_config

        Returns:
            Union[str, dict, list] -- The value of the located configuration entry.
        """
        _fcc = self.feature_column_config(_name)
        if _attribute_name in _fcc:
            return _fcc[_attribute_name]
        else:
            raise KeyError(
                "{} is not found for column {}".format(_attribute_name, _name)
            )

    def shape(self, _name: str) -> list:
        """Return the shape of column _name; throw error if not found shape info.

        Arguments:
            _name {str} -- The name of the target column

        Returns:
            list -- A list containing the shape info.
        """
        _result = self.feature_column_config_attribute(_name, "shape")
        if type(_result) != list:
            raise TypeError("shape of {} is not a list".format(_name))
        return _result

    def fc_type(self, _name: str) -> str:
        """Return the feature column type of column _name.

        Arguments:
            _name {str} -- The name of the target column.

        Returns:
            str -- Feature column type.
        """
        _result = self.feature_column_config_attribute(_name, "fcType")
        if type(_result) != str:
            raise TypeError("fcType of {} is not a string".format(_name))
        return _result

    def fc_params(self, _name: str) -> dict:
        """Return all feature column configuration for column _name.

        Arguments:
            _name {str} -- the name of the target column.

        Returns:
            dict -- A dict of located feature column configuration.
        """
        _result = self.feature_column_config_attribute(_name, "fcParam")[
            self.fc_type(_name)
        ]
        if type(_result) != dict:
            raise TypeError("fcParam of {} is not a dict".format(_name))
        return _result

    def dtype(self, _name: str) -> str:
        """Return the dtype of the target column.

        Arguments:
            _name {str} -- The name of the target column.

        Returns:
            str -- The dtype of the target column.
        """
        _dtype = self.fc_params(_name)["dtype"]
        if _dtype is None:
            raise KeyError("missing dtype in column {}".format(_name))
        if type(_dtype) is not str:
            raise TypeError("dtype for column {} is not string".format(_name))
        if _dtype not in set(self._feature_dtypes.keys()):
            raise ValueError("dtype for column {} is illegal".format(_name))
        return self._feature_dtypes[_dtype]
