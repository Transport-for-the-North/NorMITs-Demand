"""
Run the Travel Market Synthesiser
"""

import os
import pandas as pd
import importlib as ri

import normits_demand.build.tms_pathing as tms
import normits_demand.models.external_model as em
import normits_demand.models.distribution_model as dm
import normits_demand.matrices.tms_pa_to_od as pa2od
import normits_demand.reports.reports_audits as ra
from normits_demand.utils import vehicle_occupancy as vo
from normits_demand.utils import utils as nup
from normits_demand.utils import file_ops

from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.matrices import matrix_processing as mat_p


# GLOBAL VARIABLES
years = [2018]
scenario = "NTEM"
notem_iter = '4'
lu_drive = "I:/"
by_iteration = "iter3d"
fy_iteration = "iter3d"
notem_import_home = r"I:\NorMITs Demand\import\NoTEM"
# notem_export_home = r"C:\Data\Nirmal_Atkins"
notem_export_home = r"E:\NoTEM"
output_file = "%s_msoa_notem_segmented_%d_dvec.pkl"

TLD_HOME = r"I:\NorMITs Synthesiser\import\trip_length_bands"


def check_notem_run_status() -> None:
    hb_fname = output_file % ('hb', years)
    nhb_fname = output_file % ('nhb', years)
    hb_trip_ends = ['HB_Productions', 'HB_Attractions']
    nhb_trip_ends = ['NHB_Productions', 'NHB_Attractions']
    for trip in hb_trip_ends:
        file = os.path.join(notem_export_home, trip, hb_fname)
        file_ops.check_file_exists(file)
    for trip in nhb_trip_ends:
        file = os.path.join(notem_export_home, trip, nhb_fname)
        file_ops.check_file_exists(file)


class TmsParameterBuilder:
    """
    - get_model_name() module gets the model name from the user and returns model name
        It also does input validation of the user input
    - assign_param() module saves the parameters required for model runs based on the model name entered by user
        It returns a dictionary params
    """

    def get_model_name(self):
        model_list = {1: 'noham', 2: 'norms'}
        print(model_list)
        try:
            name = int(input('Choose a model name (index): '))
        except ValueError:
            print("Enter a valid index")
        else:
            if name == 1 or name == 2:
                model_name = model_list[int(name)]
                return model_name
            else:
                print("Invalid input")

    def assign_param(self, model: str):

        if model == 'noham':
            params = {
                'base_directory': 'I:/NorMITs Synthesiser',
                'model_name': 'noham',
                'iteration': 'iter_test',
                'model_zoning': 'Noham',
                'segmentation_type': 'tfn',
                'hb_output_segments': ['p', 'm'],
                'nhb_output_segments': ['area_type', 'p', 'm', 'ca', 'tp'],
                'land_use_path': 'I: / NorMITs Land Use / iter3 / land_use_output_msoa.csv',
                'control_production_to_ntem': True,
                'k_factor_path': None,
                'export_msoa_productions': False,
                'attraction_segment_type': 'ntem',
                'cjtw_path': None,
                'hb_distribution_segmentation': ['p', 'm'],
                'nhb_distribution_segments': ['p', 'm', 'tp'],
                'distribution_segmentation': ['p', 'm'],
                'distribution_function': 'ln',

                # EXTERNAL MODEL
                'tld_area': 'gb',
                'internal_tld_bands': 'p_m_standard_bands',
                'external_tld_bands': 'p_m_large_bands',
                'external_segmentation': ['p', 'm'],
                'external_export_modes': [3, 5],

                'output_modes': [3, 5],
                'non_dist_export_modes': None,
                'intrazonal_modes': [1, 2],
                'infill_modes': 6,
                'synthetic_modes': 3,
                'rail_fusion': False,
                'compile_pa': False,
                'compile_od': True,
                'vehicle_demand': True
            }
            return params
        else:
            params = {
                'base_directory': 'I:/NorMITs Synthesiser',
                'model_name': 'norms',
                'iteration': 'iter7',
                'model_zoning': 'Norms',
                'land_use_version': 3,
                'hb_output_segments': ['p', 'm'],
                'hb_attraction_weights': 'hb_attraction_weights.csv',
                'nhb_attraction_weights': 'nhb_attraction_weights.csv',
                'attraction_mode_split': 'attraction_mode_split.csv',
                'production_ntem_control': True,
                'attraction_ntem_control': True,
                'ntem_control_path': 'I:/NorMITs Synthesiser/import/ntem_constraints/ntem_pa_ave_wday_2018.csv',
                'production_k_factor_control': False,
                'production_k_factor_path': False,
                'attraction_k_factor_control': False,
                'attraction_k_factor_path': False,
                'export_msoa': True,
                'export_uncorrected': False,
                'export_lad': False,
                'export_model_zoning': True,

                # EXTERNAL MODEL
                'run_external_models': True,
                'tld_area': 'gb',
                'internal_tld_bands': 'p_m_ca_standard_bands',
                'external_tld_bands': 'p_m_ca_large_bands',
                'external_segmentation': ['p', 'm', 'ca'],
                'external_export_modes': [6],
                'non_dist_export_modes': None,

                'run_distribution': True,
                'hb_distribution_segmentation': ['p', 'm', 'ca'],
                'nhb_distribution_segmentation': ['p', 'm', 'ca', 'tp'],
                'output_modes': [6],
                'cjtw_modes': None,
                'intrazonal_modes': [1, 2],
                'infill_modes': 3,
                'synthetic_modes': 6,
                'fusion_modes': 6,
                'cost_method': 'distance',
                'distribution_tlb_area': 'north',
                'distribution_function': 'ln',
                'furness_loops': 2000,
                'fitting_loops': 100,
                'compile_pa': True,
                'pa_compilation_index': 'pa_compilation_params.csv',
                'compile_od': True,
                'od_compilation_index': 'od_compilation_params.csv',
                'disaggregate_segments': True,
                'segments_to_add': ['soc', 'ns'],
                'vehicle_demand': False,

                # PA2OD params
                'hb_p_needed': [1, 2, 3, 4, 5, 6, 7, 8],
                'nhb_p_needed': [12, 13, 14, 15, 16, 18],
                'm_needed': [6],
                'ca_needed': [1, 2],
            }
            return params


def pa_to_od(params):
    # DONT HARDCODE THIS IN OO
    out_home = os.path.join(
        params['base_directory'],
        params['model_name'],
        params['iteration'],
        'Distribution Outputs',
    )
    external_out = os.path.join(out_home, 'External Distributions')
    internal_out = os.path.join(out_home, '24hr PA Distributions')
    gb_out = os.path.join(out_home, '24hr PA Distributions - GB')
    od_out = os.path.join(out_home, 'OD Matrices')
    # Combine internal and external


    # Set up the segmentation params
    seg_level = 'tms'
    seg_params = {
        'p_needed': params['hb_p_needed'],
        'm_needed': params['m_needed'],
        'ca_needed': params['ca_needed'],
    }

    # Convert HB to OD via tour proportions
    pa2od.build_od_from_fh_th_factors(
        pa_import=gb_out,
        od_export=od_out,
        fh_th_factors_dir=self.imports['post_me_fh_th_factors'],
        years_needed=[2018],
        seg_level=seg_level,
        seg_params=seg_params
    )

    # Convert NHB to tp split via factors
    nhb_seg_params = seg_params.copy()
    nhb_seg_params['p_needed'] = params['nhb_p_needed']

    mat_p.nhb_tp_split_via_factors(
        import_dir=gb_out,
        export_dir=od_out,
        import_matrix_format='pa',
        export_matrix_format='od',
        tour_proportions_dir=self.imports['post_me_tours'],
        model_name=params['model_name'],
        years_needed=[2018],
        **nhb_seg_params,
    )


if __name__ == '__main__':
    # Checks whether NoTEM trip end runs are completed
    #check_notem_run_status()

    config_path = 'I:/NorMITs Synthesiser/config/'
    tmsparam = TmsParameterBuilder()

    # Gets model name from the user using the module
    model_name = tmsparam.get_model_name()

    # Assigns the various model parameters required based on model_name
    params = tmsparam.assign_param(model_name)

    # TODO: Building loose folders for External model paths
    tms_run = tms.TMSPathing(config_path,
                             params)

    # Check status of lookup folder
    tms_run.lookups = tms_run.lookup_audit()

    # Update project status
    # BACKLOG: Project status
    tms_run.project_status = tms_run.project_check()


    # TODO: Define init params

    # Setup up TLD paths
    # Path tlb folder
    tld_dir = os.path.join(TLD_HOME, params['tld_area'])
    internal_tld_path = os.path.join(tld_dir, params['internal_tld_bands'])
    external_tld_path = os.path.join(tld_dir, params['external_tld_bands'])

    # # Run HB external model
    # ext = em.ExternalModel(
    #     config_path,
    #     params,
    # )
    #
    # hb_ext_out = ext.run(
    #     trip_origin='hb',
    #     cost_type='24hr',
    #     internal_tld_path=internal_tld_path,
    #     external_tld_path=external_tld_path,
    # )
    #
    # nhb_ext_out = ext.run(
    #     trip_origin='nhb',
    #     cost_type='24hr',
    #     internal_tld_path=internal_tld_path,
    #     external_tld_path=external_tld_path,
    # )
    #
    # print("External Model DOne")
    # exit()

    dist = dm.DistributionModel(
        config_path,
        params)

    dist.run_distribution_model(
        file_drive=params['base_directory'],
        model_name=params['model_name'],
        iteration=params['iteration'],
        tlb_area='north',
        segmentation='tfn',
        distribution_segments=params['hb_distribution_segmentation'],
        dist_function=params['distribution_function'],
        trip_origin='hb',
        cost_type='24hr',
        furness_loops=1999,
        fitting_loops=100,
        iz_cost_infill=.5,
        export_modes=params['synthetic_modes'],
        mp_threads=0,
    )

    dist.run_distribution_model(
        file_drive=params['base_directory'],
        model_name=params['model_name'],
        iteration=params['iteration'],
        tlb_area='north',
        segmentation='tfn',
        distribution_segments=params['nhb_distribution_segmentation'],
        dist_function=params['distribution_function'],
        trip_origin='nhb',
        cost_type='tp',
        furness_loops=1999,
        fitting_loops=100,
        iz_cost_infill=.5,
        export_modes=params['synthetic_modes'],
        verbose=True,
        mp_threads=-2,
    )

    # PA RUN REPORTS
    # Matrix Trip ENd totals
    # Sector Reports Dvec style
    # TLD curve
    #   single mile bands - p/m (ca ) segments full matrix
    #   NorMITs Vis

    pa_to_od()
    # DO PA TO OD

    # OD RUN REPORTS

    # Copy across and rename pa
    # TODO: Should do these at the start and have as references
    pa_folder = os.path.join(params['base_directory'],
                                         'NorMITs Synthesiser',
                                         params['model_name'],
                                         params['iteration'],
                                         'Distribution Outputs',
                                         'PA Matrices')

    od_folder = os.path.join(params['base_directory'],
                             'NorMITs Synthesiser',
                             params['model_name'],
                             params['iteration'],
                             'Distribution Outputs',
                             'OD Matrices')

    nhb_pa = [x for x in [x for x in os.listdir(pa_folder) if '.csv' in x] if 'nhb' in x]

    # REname PA to OD
    for nhb_mat in nhb_pa:
        od_mat = nhb_mat.replace('pa', 'od')
        od_out = pd.read_csv(os.path.join(pa_folder, nhb_mat))
        print(od_mat)
        od_out.to_csv(os.path.join(od_folder, od_mat), index=False)

    lookup_folder = os.path.join(params['base_directory'],
                                 params['model_name'],
                                 'Model Zone Lookups')
    o_paths_pa = os.path.join(params['base_directory'],
                              params['model_name'],
                              params['iteration'],
                              'Distribution Outputs',
                              'PA Matrices')
    o_paths_c_pa_export = os.path.join(params['base_directory'],
                                       params['model_name'],
                                       params['iteration'],
                                       'Distribution Outputs',
                                       'Compiled PA Matrices')



    # COMPILE TO ASSIGNMENT MODEL SEGMENTATION
