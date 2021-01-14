# -*- coding: utf-8 -*-
"""
    Module for running the elasticity model from command line,
    which will eventually be supersceded by a wrapper script
    which will run the whole ESF process.
"""

##### IMPORTS #####
# Standard imports
import sys
from pathlib import Path
from configparser import ConfigParser, NoSectionError
from typing import Tuple, List, Dict

# Third party imports

# Local imports
# Appending to path to avoid import errors
sys.path.append(str(Path(__file__).parent.parent))
from elasticity_calcs.elasticity_model import ElasticityModel


##### CONSTANTS #####
CONFIG_FILE = "elasticity_calcs/config.txt"


##### FUNCTIONS #####
def get_inputs() -> Tuple[Dict[str, Path], Dict[str, Path], Path, List[str]]:
    """Read config file to get the input and output paths and information.

    The config file requires the following sections, with
    the respective values in each:
    - input_folders: elasticity, translation, rail_demand, car_demand,
        rail_costs, car_costs
    - input_files: gc_parameters, cost_changes
    - other: output_folder, years

    Returns
    -------
    Dict[str, Path]
        Paths to the input folders.
    Dict[str, Path]
        Paths to the input files.
    Path
        Path to the ouput folder.
    List[str]
        List of the model years.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist.
    NoSectionError
        If there is a section missing from the config file.
    """
    config = ConfigParser()
    path = Path(CONFIG_FILE)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find config file: {path}")
    config.read(path)

    missing_sections = []
    for s in ("input_folders", "input_files", "other"):
        if s not in config.sections():
            missing_sections.append(s)
    if missing_sections:
        raise NoSectionError(missing_sections)

    input_folders = {k: Path(p) for k, p in config.items("input_folders")}
    input_files = {k: Path(p) for k, p in config.items("input_files")}
    output_folder = Path(config.get("other", "output_folder"))
    years = config.get("other", "years").split(",")
    return input_folders, input_files, output_folder, years


def main():
    """Runs the elasticity model using parameters from the config file."""
    input_folders, input_files, output_folder, years = get_inputs()
    elast_model = ElasticityModel(
        input_folders, input_files, output_folder, years
    )
    elast_model.apply_all()


##### MAIN #####
if __name__ == "__main__":
    main()
