# -*- coding: utf-8 -*-
"""
Created on: 09/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Classes which build all the paths for TMS model inputs outputs
"""
# Built-Ins
from abc import ABC
from abc import abstractmethod

from typing import Dict

# Third Party

# Local Imports
import normits_demand as nd


class TMSImportClassBase(ABC):
    """Abstract Class defining how the import paths class for TMS should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions to pick up new import files.
    """

    @abstractmethod
    def generate_external_model_imports(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        raise NotImplementedError(
            "generate_external_model_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )

    @abstractmethod
    def generate_gravity_model_imports(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        raise NotImplementedError(
            "generate_gravity_model_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )


class TMSExportPaths:
    # TODO(BT): Finalise TMS exports structure

    def __init__(self):
        # We would assign export paths here
        pass
