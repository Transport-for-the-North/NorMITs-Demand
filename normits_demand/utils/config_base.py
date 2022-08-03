# -*- coding: utf-8 -*-
"""Base config class for storing and reading parameters for any NorMITs demand script."""

##### IMPORTS #####
import json
import pathlib
from typing import TypeVar

import pydantic
import strictyaml


##### CONSTANTS #####
TConfig = TypeVar("TConfig", bound="BaseConfig")

##### CLASSES #####
class BaseConfig(pydantic.BaseModel):
    r"""Base class for storing model parameters.

    Contains functionality for reading / writing parameters to
    config files in the YAML format.

    See Also
    --------
    [pydantic docs](https://pydantic-docs.helpmanual.io/):
        for more information about using pydantic's model classes.
    `pydantic.BaseModel`: which handles converting data to Python types.
    `pydantic.validator`: which allows additional custom validation methods.

    Examples
    --------
    >>> import pathlib

    >>> import normits_demand as nd
    >>> from normits_demand.utils import config_base

    >>> class ExampleParameters(config_base.BaseConfig):
    ...    import_folder: pathlib.Path
    ...    notem_iteration: str
    ...    scenario: nd.Scenario

    >>> parameters = ExampleParameters(
    ...    import_folder="Test Folder",
    ...    notem_iteration="1.0",
    ...    scenario=nd.Scenario.NTEM,
    ... )
    >>> parameters
    ExampleParameters(import_folder=WindowsPath('Test Folder'), notem_iteration='1.0', scenario=<Scenario.NTEM: 'NTEM'>)

    >>> parameters.to_yaml()
    'import_folder: Test Folder\nnotem_iteration: 1.0\nscenario: NTEM\n'

    >>> yaml_text = '''
    ... import_folder: Test YAML Folder
    ... notem_iteration: 1.0
    ... scenario: NTEM
    ... '''
    >>> ExampleParameters.from_yaml(yaml_text)
    ExampleParameters(import_folder=WindowsPath('Test YAML Folder'), notem_iteration='1.0', scenario=<Scenario.NTEM: 'NTEM'>)
    """

    @classmethod
    def from_yaml(cls, text: str):
        """Parse class attributes from YAML `text`.

        Parameters
        ----------
        text : str
            YAML formatted string, with parameters for
            the class attributes.

        Returns
        -------
        Instance of self
            Instance of class with attributes filled in from
            the YAML data.
        """
        data = strictyaml.load(text).data
        return cls.parse_obj(data)

    @classmethod
    def load_yaml(cls, path: pathlib.Path):
        """Read YAML file and load the data using `from_yaml`.

        Parameters
        ----------
        path : pathlib.Path
            Path to YAML file containing parameters.

        Returns
        -------
        Instance of self
            Instance of class with attributes filled in from
            the YAML data.
        """
        with open(path, "rt") as file:
            text = file.read()
        return cls.from_yaml(text)

    def to_yaml(self) -> str:
        """Convert attributes from self to YAML string.

        Returns
        -------
        str
            YAML formatted string with the data from
            the class attributes.
        """
        # Use pydantic to convert all types to json compatiable,
        # then convert this back to a dictionary to dump to YAML
        json_dict = json.loads(self.json())
        return strictyaml.as_document(json_dict).as_yaml()

    def save_yaml(self, path: pathlib.Path) -> None:
        """Write data from self to a YAML file.

        Parameters
        ----------
        path : pathlib.Path
            Path to YAML file to output.
        """
        with open(path, "wt") as file:
            file.write(self.to_yaml())


##### FUNCTIONS #####
