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
                'external_tlb_area': 'gb',
                'external_tlb_name': 'external_ph_segments',
                'external_segmentation': ['p', 'm'],
                'external_export_modes': [3],
                'output_modes': 3,
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
                'run_external_models': True,
                'external_tlb_area': 'gb',
                'external_tlb_name': 'standard_plus_ca_segments',
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
            }
            return params


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

    # #Run HB external model
    # ext = em.ExternalModel(
    #     config_path,
    #     params)
    #
    # hb_ext_out = ext.run(
    #     trip_origin='hb',
    #     cost_type='24hr',
    # )
    # nhb_ext_out = ext.run(
    #     trip_origin='nhb',
    #     cost_type='24hr',
    # )

    dist = dm.DistributionModel(
        config_path,
        params)

    int_hb = dist.run_distribution_model(
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

    print("DONE!")
    exit()

    int_nhb = dist.run_distribution_model(
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
        mp_threads=-2)


    # Compile tp pa
    pa2od.build_tp_pa(file_drive=params['base_directory'],
                      model_name=params['model_name'],
                      iteration=params['iteration'],
                      distribution_segments=params['hb_output_segments'],
                      internal_input='synthetic',
                      external_input='synthetic',
                      write_modes=params['output_modes'],
                      arrivals=False,
                      write=True)

    # *Compile tp pa @ 24hr
    pa2od.build_tp_pa(file_drive=params['base_directory'],
                      model_name=params['model_name'],
                      iteration=params['iteration'],
                      distribution_segments=params['hb_output_segments'],
                      internal_input='synthetic',
                      external_input='synthetic',
                      write_modes=params['output_modes'],
                      arrivals=False,
                      export_24hr=True,
                      write=True)

    # Build any non dist tps as required
    pa2od.build_tp_pa(file_drive=params['base_directory'],
                      model_name=params['model_name'],
                      iteration=params['iteration'],
                      distribution_segments=params['hb_output_segments'],
                      internal_input='non_dist',
                      external_input='non_dist',
                      write_modes=[6],
                      arrivals=False,
                      export_24hr=False,
                      write=True)

    # Build OD
    od_audit = pa2od.build_od(file_drive=params['base_directory'],
                              model_name=params['model_name'],
                              iteration=params['iteration'],
                              distribution_segments=params['hb_output_segments'],
                              internal_input='synthetic',
                              external_input='synthetic',
                              phi_type='fhp_tp',
                              export_modes=[3],
                              write=True)

    # Export other modes from non-compiled
    pa2od.build_od(file_drive=params['base_directory'],
                   model_name=params['model_name'],
                   iteration=params['iteration'],
                   distribution_segments=params['hb_output_segments'],
                   internal_input='non_dist',
                   external_input='non_dist',
                   phi_type='fhp_tp',
                   export_modes=[6],  # Define as parameter
                   write=True)

    print(od_audit)

    pa2od.compile_nhb_pa(file_drive=params['base_directory'],
                         model_name=params['model_name'],
                         iteration=params['iteration'],
                         distribution_segments=['p', 'm', 'tp'],
                         internal_input='synthetic',
                         external_input='synthetic',
                         export_modes=[3],
                         write=True)

    # TODO: This needs some attention to work with non-dist segs
    pa2od.compile_nhb_pa(file_drive=params['base_directory'],
                         model_name=params['model_name'],
                         iteration=params['iteration'],
                         distribution_segments=['p', 'm', 'tp'],
                         internal_input='non_dist',
                         external_input='non_dist',
                         export_modes=[1, 2, 5],
                         write=True)

    # Copy across and rename pa
    # TODO: Should do these at the start and have as references
    pa_folder = od_folder = os.path.join(params['base_directory'],
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

    nhb_pa = [x for x in [x for x in os.listdir(
        pa_folder) if '.csv' in x] if 'nhb' in x]

    for nhb_mat in nhb_pa:
        od_mat = nhb_mat.replace('pa', 'od')
        od_out = pd.read_csv(os.path.join(pa_folder, nhb_mat))
        print(od_mat)
        od_out.to_csv(os.path.join(od_folder, od_mat), index=False)

    # Run reports
    ra.distribution_report(params['base_directory'],
                           params['model_name'],
                           params['iteration'],
                           params['hb_output_segments'],
                           distributions='Distribution Outputs/OD Matrices',
                           matrix_format='wide',
                           report_tp='24hr',
                           internal_reports=True,
                           write=True)

    # PA distribution reports (production audit)
    # TODO: Get to work with new outputs
    ra.distribution_report(params['base_directory'],
                           params['model_name'],
                           params['iteration'],
                           params['hb_output_segments'],
                           distributions='Distribution Outputs/PA Matrices',
                           matrix_format='wide',
                           report_tp='24hr',
                           internal_reports=True,
                           write=True)

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

    # Import compilation params
    pa_compilation_params, od_compilation_params = nup.get_compilation_params(lookup_folder)

    if params['compile_pa']:
        compiled_pa_matrices = nup.compile_pa(o_paths_pa,
                                              (lookup_folder +
                                               '/' +
                                               pa_compilation_params))  # Split time = true

        # Export compiled PA
        for mat in compiled_pa_matrices:
            for key, value in mat.items():
                print(key)
                c_pa_out = (o_paths_c_pa_export +
                            '/' +
                            key +
                            '.csv')
                value.to_csv(c_pa_out, index=False)

        # Reports on compiled PA and OD
        ra.distribution_report(params['base_directory'],
                               params['model_name'],
                               params['iteration'],
                               ['p', 'm', 'ca'],
                               distributions='Distribution Outputs/Compiled PA Matrices',
                               matrix_format='long',
                               report_tp='tp',
                               internal_reports=False,
                               write=True)

    # TODO: Export factors
    # TODO: Export from function
    if params['compile_od']:
        compiled_od_matrices = nup.compile_od(
            od_folder=os.path.join(params['base_directory'],
                                   'NorMITs Synthesiser',
                                   params['model_name'],
                                   params['iteration'],
                                   'Distribution Outputs',
                                   'OD Matrices'),
            write_folder=os.path.join(params['base_directory'],
                                      'NorMITs Synthesiser',
                                      params['model_name'],
                                      params['iteration'],
                                      'Distribution Outputs',
                                      'Compiled OD Matrices'),
            compile_param_path=os.path.join(lookup_folder,
                                            params['model_name'].lower() +
                                            '_od_matrix_params.csv'),
            build_factor_pickle=True)

        # Build compiled od for non dist segs
        # TODO: if statement
        nup.compile_od(
            od_folder=os.path.join(params['base_directory'],
                                   'NorMITs Synthesiser',
                                   params['model_name'],
                                   params['iteration'],
                                   'Distribution Outputs',
                                   'OD Matrices Non Dist'),
            write_folder='C:/Users/genie/Documents/Nelum frh',  # Write fast
            compile_param_path=os.path.join(
                lookup_folder,
                params['model_name'].lower() +
                '_non_dist_od_frh_matrix_params.csv'),
            build_factor_pickle=False)

        # Run audit on compiled OD
        ra.distribution_report(params['base_directory'],
                               params['model_name'],
                               params['iteration'],
                               params['hb_output_segments'],
                               distributions='Distribution Outputs/Compiled OD Matrices',
                               matrix_format='wide',
                               report_tp='tp',
                               internal_reports=False,
                               write=True)

        # *Translate trips to vehicles

    # Export should be:
    os.path.join(params['base_directory'],
                 'NorMITs Synthesiser',
                 params['model_name'],
                 params['iteration'],
                 'PCU Outputs',
                 'PCU OD Matrices')

    # Convert to vehicles
    if params['vehicle_demand']:
        vo.people_vehicle_conversion(import_folder=os.path.join(params['base_directory'],
                                                                'NorMITs Synthesiser',
                                                                params['model_name'],
                                                                params['iteration'],
                                                                'Distribution Outputs',
                                                                'Compiled OD Matrices'),
                                     export_folder='D:/',
                                     mode='3',
                                     method='to_vehicles',
                                     hourly_average=True,
                                     out_format='long',
                                     rounding_factor=None,
                                     header=False,
                                     write=True)
        # TODO: add 3 sectore report call
