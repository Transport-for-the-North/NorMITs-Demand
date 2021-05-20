# -*- coding: utf-8 -*-
"""
    Module for running the elasticity model from command line,
    which will eventually be supersceded by a wrapper script
    which will run the whole ESF process.
"""

##### imports #####
# Standard imports
import sys
import os
import configparser

from pathlib import Path
from typing import Tuple, List, Dict

# Third party imports

# Local imports
import normits_demand as nd

from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts
from normits_demand.utils import file_ops


##### CONSTANTS #####
CONFIG_FILE = "config/setup/elasticity_config.txt"


##### FUNCTIONS #####
def get_inputs() -> Tuple[Dict[str, Path],
                          Dict[str, Path],
                          Dict[str, Path],
                          List[str]]:
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


def _create_input_paths(noham_efs, norms_efs, use_bespoke_zones):
    # Set up some paths
    if use_bespoke_zones:
        rail_demand = norms_efs.exports['pa_24_bespoke']
        car_demand = noham_efs.exports['pa_24_bespoke']
    else:
        rail_demand = norms_efs.exports['pa_24']
        car_demand = noham_efs.exports['pa_24']

    # Build the dictionary
    input_dict = {
        'elasticity': os.path.join(noham_efs.imports['home'], 'elasticities'),
        'translation': noham_efs.imports['zone_translation']['no_overlap'],
        'rail_demand': rail_demand,
        'car_demand': car_demand,
        'rail_costs': os.path.join(norms_efs.imports['costs'], 'elasticity_model_format'),
        'car_costs': os.path.join(noham_efs.imports['costs'], 'elasticity_model_format'),
    }

    return input_dict


def _create_input_files(efs, scenario):
    scenario_cost_home = os.path.join(efs.imports['home'], 'elasticities', scenario)

    # Build the dictionary
    input_dict = {
        'gc_parameters': os.path.join(scenario_cost_home, 'gc_parameters-uzc.csv'),
        'cost_changes': os.path.join(scenario_cost_home, 'cost_changes-uzc-temp.csv'),
    }

    return input_dict


def _create_output_files(noham_efs, norms_efs, iteration, scenario):
    mat_dir_name = '24hr PA Matrices Elasticity'

    others = os.path.join(
        noham_efs.exports['base'],
        'elasticity',
        iteration,
        scenario,
    )

    # Build the dictionary
    output_dict = {
        'car': os.path.join(noham_efs.exports['mat_home'], mat_dir_name),
        'rail': os.path.join(norms_efs.exports['mat_home'], mat_dir_name),
        'others': others,

    }

    # Make sure all the paths exits
    for _, v in output_dict.items():
        file_ops.create_folder(v, verbose=False)

    return output_dict


def _create_other_args(years):
    return {'years': ','.join([str(x) for x in years])}


def initialise():
    # Where to write the config file
    fname = 'elasticity_config.txt'
    config_path = os.path.join(os.getcwd(), 'config', 'setup', fname)

    # Controls I/O
    scenario = efs_consts.SC04_UZC
    iter_num = '3g'
    import_home = "I:/"
    export_home = "I:/"

    years = [2033, 2040, 2050]
    use_bespoke_zones = False

    # ## INITIALISE EFS TO GET PATHS ## #
    efs_params = {
        'iter_num': iter_num,
        'scenario_name': scenario,
        'import_home': import_home,
        'export_home': export_home,
        'verbose': False,
    }

    noham_efs = nd.ExternalForecastSystem(model_name='noham', **efs_params)
    norms_efs = nd.ExternalForecastSystem(model_name='norms', **efs_params)

    # Build the output paths
    input_paths = _create_input_paths(noham_efs, norms_efs, use_bespoke_zones)
    input_files = _create_input_files(noham_efs, scenario)
    output_paths = _create_output_files(noham_efs, norms_efs, iter_num, scenario)
    other_args = _create_other_args(years)

    # ## WRITE OUR THE CONFIG FILE ## #
    # Create the object
    config = configparser.ConfigParser()
    config['input_folders'] = input_paths
    config['input_files'] = input_files
    config['output_folders'] = output_paths
    config['other'] = other_args

    # Write out to disk
    with open(config_path, 'w') as f:
        config.write(f)


def main():
    """Runs the elasticity model using parameters from the config file."""

    # TODO(BT): Add as a param to the elasticity
    base_year = 2018

    input_folders, input_files, output_folder, years = get_inputs()
    elast_model = nd.ElasticityModel(
        input_folders=input_folders,
        input_files=input_files,
        output_folders=output_folder,
        output_years=years,
        base_year=base_year,
    )
    elast_model.apply_all()


if __name__ == "__main__":
    # initialise()
    main()
