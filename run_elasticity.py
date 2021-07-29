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

from normits_demand.matrices import matrix_processing as mat_p


##### CONSTANTS #####
CONFIG_FILE = "config/setup/elasticity_config.txt"


##### FUNCTIONS #####
def get_inputs() -> Tuple[Dict[str, Path],
                          Dict[str, Path],
                          Dict[str, Path],
                          int,
                          List[str],
                          ]:
    """Read config file to get the input and output paths and information.

    The config file requires the following sections, with
    the respective values in each:
    - input_folders: elasticity, translation, rail_demand, car_demand,
        rail_costs, car_costs
    - input_files: gc_parameters, cost_changes
    - output_folders: car, rail, others
    - other: output_folder, future_years

    Returns
    -------
    Dict[str, Path]
        Paths to the input folders.
    Dict[str, Path]
        Paths to the input files.
    Dict[str, Path]
        Paths to the ouput folders.
    List[str]
        List of the model future_years.

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

    # Extract the groups of inputs
    input_folders = {k: Path(p) for k, p in config.items("input_folders")}
    input_files = {k: Path(p) for k, p in config.items("input_files")}
    output_folders = {k: Path(p) for k, p in config.items("output_folders")}

    # Extract the individual inputs
    base_year = config.get("other", "base_year")
    future_years = config.get("other", "future_years").split(",")

    return input_folders, input_files, output_folders, base_year, future_years


def _create_input_paths(noham_efs, norms_efs, use_bespoke_zones, use_wfh_adj):
    if use_wfh_adj and use_bespoke_zones:
        raise ValueError(
            "Cannot use both WFH adjustments and bespoke zone outputs!"
        )

    # Set up some paths
    if use_bespoke_zones:
        rail_demand = norms_efs.exports['pa_24_bespoke']
        car_demand = noham_efs.exports['pa_24_bespoke']
    elif use_wfh_adj:
        rail_demand = norms_efs.exports['pa_24_wfh']
        car_demand = noham_efs.exports['pa_24_wfh']
    else:
        rail_demand = norms_efs.exports['pa_24']
        car_demand = noham_efs.exports['pa_24']

    # Build the dictionary
    input_dict = {
        'import_home': os.path.join(noham_efs.imports['home'], 'elasticities'),
        'translation_home': noham_efs.imports['zone_translation']['one_to_one'],
        'rail_demand_import': os.path.join(rail_demand, 'internal'),
        'car_demand_import': os.path.join(car_demand, 'internal'),
        'rail_costs_import': os.path.join(norms_efs.imports['costs'], 'elasticity_model_format'),
        'car_costs_import': os.path.join(noham_efs.imports['costs'], 'elasticity_model_format'),
    }

    return input_dict


def _create_input_files(efs, scenario):
    scenario_cost_home = os.path.join(
        efs.imports['home'],
        'scenarios',
        scenario,
        'elasticity',
    )

    # Build the dictionary
    input_dict = {
        'vot_voc_values_path': os.path.join(scenario_cost_home, 'vot_voc_values.csv'),
        'cost_adjustments_path': os.path.join(scenario_cost_home, 'cost_adjustments.csv'),
    }

    return input_dict


def _create_output_files(noham_efs, norms_efs, iteration, scenario):

    others = os.path.join(
        noham_efs.exports['base'],
        'elasticity',
        'iter%s' % iteration,
        scenario,
    )

    # Build the dictionary
    output_dict = {
        'car_demand_export': os.path.join(noham_efs.exports['pa_24_elast'], 'internal'),
        'rail_demand_export': os.path.join(norms_efs.exports['pa_24_elast'], 'internal'),
        'other_demand_export': others,
        'reports_export': os.path.join(others, 'Reports'),
    }

    # Make sure all the paths exits
    for _, v in output_dict.items():
        file_ops.create_folder(v, verbose=False)

    return output_dict


def _create_other_args(base_year, future_years):
    return {
        'base_year': base_year,
        'future_years': ','.join([str(x) for x in future_years])
    }


def initialise(scenario,
               iter_num,
               import_home,
               export_home,
               base_year,
               future_years,
               use_bespoke_zones,
               use_wfh_adj,
               ):
    # Where to write the config file
    fname = 'elasticity_config.txt'
    config_path = os.path.join(os.getcwd(), 'config', 'setup', fname)

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
    input_paths = _create_input_paths(noham_efs, norms_efs, use_bespoke_zones, use_wfh_adj)
    input_files = _create_input_files(noham_efs, scenario)
    output_paths = _create_output_files(noham_efs, norms_efs, iter_num, scenario)
    other_args = _create_other_args(base_year, future_years)

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


def run_elasticity(base_year):
    """Runs the elasticity model using parameters from the config file."""

    input_folders, input_files, output_folders, base_year, future_years = get_inputs()
    elast_model = nd.ElasticityModel(
        base_year=base_year,
        future_years=future_years,
        **input_folders,
        **input_files,
        **output_folders,
    )
    elast_model.apply_all_MP()


def merge_internal_external(scenario,
                            iter_num,
                            import_home,
                            export_home,
                            base_year,
                            future_years,
                            use_bespoke_zones,
                            use_wfh_adj,
                            ):
    # Init
    valid_ftypes = ['.csv', consts.COMPRESSION_SUFFIX]

    # Set up some paths
    if use_bespoke_zones:
        external_key = 'pa_24_bespoke'
    elif use_wfh_adj:
        external_key = 'pa_24_wfh'
    else:
        external_key = 'pa_24'

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

    # Combine the internal and external trips
    for name, mode_efs in zip(['noham', 'norms'], [noham_efs, norms_efs]):
        # ## COPY OVER 2018 INTERNAL ## #
        by_str = '_yr%s_' % base_year

        # Get paths for just base_year matrices
        pre_elast_internal = os.path.join(mode_efs.exports[external_key], 'internal')
        internal_by = file_ops.list_files(pre_elast_internal, ftypes=valid_ftypes)
        internal_by = [x for x in internal_by if by_str in x]

        file_ops.copy_files(
            src_dir=pre_elast_internal,
            dst_dir=os.path.join(mode_efs.exports['pa_24_elast'], 'internal'),
            filenames=internal_by,
        )

        # Set up the pbar
        pbar_kwargs = {
            'desc': f"Recombining {name} int/ext matrices",
            'colour': 'cyan',
        }

        # Recombine internal and external
        mat_p.recombine_internal_external(
            internal_import=os.path.join(mode_efs.exports['pa_24_elast'], 'internal'),
            external_import=os.path.join(mode_efs.exports[external_key], 'external'),
            full_export=mode_efs.exports['pa_24_elast'],
            years=[base_year] + future_years,
            force_compress_out=True,
            pbar_kwargs=pbar_kwargs
        )


def main():
    # Controls I/O
    scenario = efs_consts.SC04_UZC
    iter_num = '3j'
    import_home = "I:/"
    export_home = "E:/"

    # TODO(BT): Add as a param to the elasticity
    base_year = 2018
    # future_years = [2033, 2040, 2050]
    future_years = [2050]
    use_bespoke_zones = False
    use_wfh_adj = True

    initialise(
        scenario=scenario,
        iter_num=iter_num,
        import_home=import_home,
        export_home=export_home,
        base_year=base_year,
        future_years=future_years,
        use_bespoke_zones=use_bespoke_zones,
        use_wfh_adj=use_wfh_adj,
    )

    run_elasticity(base_year)

    # merge_internal_external(
    #     scenario=scenario,
    #     iter_num=iter_num,
    #     import_home=import_home,
    #     export_home=export_home,
    #     base_year=base_year,
    #     future_years=future_years,
    #     use_bespoke_zones=use_bespoke_zones,
    #     use_wfh_adj=use_wfh_adj,
    # )


if __name__ == "__main__":
    main()
