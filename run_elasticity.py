# -*- coding: utf-8 -*-
"""
    Module for running the elasticity model from command line,
    which will eventually be supersceded by a wrapper script
    which will run the whole ESF process.
"""

##### imports #####
# Standard imports
import sys
import configparser
from pathlib import Path
from typing import Tuple, List, Dict

# Third party imports

# Local imports
from normits_demand import elasticity_model as em


##### CONSTANTS #####
CONFIG_FILE = "config/setup/elasticity_config.txt"


##### FUNCTIONS #####
def get_inputs() -> Tuple[
    Dict[str, Path],
    Dict[str, Path],
    Dict[str, Path],
    List[str],
]:
    """Read config file to get the input and output paths and information.

    The config file requires the following sections, with
    the respective values in each:
    - input_folders: elasticity, translation, rail_demand, car_demand,
        rail_costs, car_costs
    - input_files: gc_parameters, cost_changes
    - output_folders: car, rail, others
    - other: output_folder, years

    Returns
    -------
    Dict[str, Path]
        Paths to the input folders.
    Dict[str, Path]
        Paths to the input files.
    Dict[str, Path]
        Paths to the ouput folders.
    List[str]
        List of the model years.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist.
    configparser.NoSectionError
        If there is a section missing from the config file.
    """
    config = configparser.ConfigParser()
    path = Path(CONFIG_FILE)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find config file: {path}")
    config.read(path)

    missing_sections = []
    for s in ("input_folders", "input_files", "output_folders", "other"):
        if s not in config.sections():
            missing_sections.append(s)
    if missing_sections:
        raise configparser.NoSectionError(missing_sections)

    input_folders = {k: Path(p) for k, p in config.items("input_folders")}
    input_files = {k: Path(p) for k, p in config.items("input_files")}
    output_folders = {k: Path(p) for k, p in config.items("output_folders")}
    years = config.get("other", "years").split(",")
    return input_folders, input_files, output_folders, years


def main():
    """Runs the elasticity model using parameters from the config file."""
    input_folders, input_files, output_folder, years = get_inputs()
    elast_model = em.ElasticityModel(
        input_folders, input_files, output_folder, years
    )
    elast_model.apply_all()


##### MAIN #####
if __name__ == "__main__":
    main()
