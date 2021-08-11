"""
Run the Travel Market Synthesiser
"""

import os
import pandas as pd
import importlib as ri

import normits_demand.build.tms_pathing as tms
import normits_demand.models.production_model as pm
import normits_demand.models.attraction_model as am
import normits_demand.models.external_model as em
import normits_demand.models.distribution_model as dm
from normits_demand import version


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

    def assign_param(self):

        if model_name == 'noham':
            params = {
                'base_directory': 'Y: /',
                'iteration': 'iter8c',
                'segmentation_type': 'tfn',
                'hb_output_segments': ['p', 'm'],
                'nhb_output_segments': ['area_type', 'p', 'm', 'ca', 'tp'],
                'land_use_path': 'Y: / NorMITs Land Use / iter3 / land_use_output_msoa.csv',
                'control_production_to_ntem': True,
                'k_factor_path': None,
                'export_msoa_productions': False,
                'attraction_segment_type': 'ntem',
                'cjtw_path': None,
                ##can be moved
                'hb_distribution_segments': ['p', 'm'],
                'nhb_distribution_segments': ['p', 'm', 'tp'],
                'output_modes': 3,
                'rail_fusion': False,
                'compile_pa': False,
                'compile_od': True,
                'vehicle_demand': True
            }
            return params
        else:
            params = {
                'base_directory': 'I:/NorMITs Synthesiser',
                'iteration': 'iter6',
                'model_zoning': 'Norms',
                'land_use_version': 3,
                'resi_land_use_path': 'Y:/NorMITs Land Use/iter3b/outputs/land_use_output_safe_msoa.csv',
                'non_resi_land_use_path': 'Y:/NorMITs Land Use/iter3b/outputs/land_use_2018_emp.csv',
                'land_use_zoning': 'MSOA',
                'run_trip_ends': True,
                'hb_trip_rates': 'tfn_hb_trip_rates_18_0620.csv',
                'hb_time_split': 'tfn_hb_time_split_18_0620.csv',
                'hb_ave_time_split': 'hb_ave_time_split.csv',
                'hb_mode_split': 'tfn_hb_mode_split_18_0620.csv',
                'nhb_trip_rates': 'tfn_nhb_ave_wday_trip_rates_18.csv',
                'nhb_time_split': 'tfn_nhb_ave_wday_time_split_18.csv',
                'nhb_mode_split': 'tfn_nhb_ave_wday_mode_split_18.csv',
                'hb_trip_end_segmentation': ['p, m, area_type, ca, soc, ns, g'],
                'nhb_trip_end_segmentation': ['p, m, area_type, ca, soc, ns, g'],
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
                'external_export_modes': 6,
                'non_dist_export_modes': None,
                'run_distribution': True,
                'distribution_segmentation': ['p', 'm', 'ca'],
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

    config_path = 'I:/NorMITs Synthesiser/config/'

    # Ask user which config file to use
    # params_file = prj.select_params_file(config_path)
    """
    Gets model name from the user using the module
    """
    model_name = TmsParameterBuilder.get_model_name()
    """
    Assigns the various model parameters required based on model_name
    """
    params_file = TmsParameterBuilder.assign_param()

    # TODO: Building loose folders for External model paths
    tms_run = tms.TMSPathing(config_path,
                             params_file)

    # Check status of lookup folder
    tms_run.lookups = tms_run.lookup_audit()

    # BACKLOG: Project status
    tms_run.project_status = tms_run.project_check()

    # BACKLOG: Do this stuff based on the project status
    p = pm.ProductionModel(config_path, params_file)

    hb_p_out = p.run_hb(verbose=True)
    p.ping_outpath()

    a = am.AttractionModel(config_path, params_file)
    hb_a_out = a.run(trip_origin='hb',
                     control_to_productions=True,
                     productions_path=p.export['in_hb'])
    a.ping_outpath()

    nhb_p_out = p.run_nhb(
        attraction_vector=a.export['in_hb'])
    p.ping_outpath()

    nhb_a_out = a.run(trip_origin='nhb',
                      control_to_productions=True,
                      productions_path=p.export['in_nhb'])

    # Delete trip end models
    del p, a
    # Update project status
    # BACKLOG: Project status

    # TODO: Define init params

    # Run HB external model
    ext = em.ExternalModel(
        config_path,
        params_file)
    hb_ext_out = ext.run(
        trip_origin='hb',
        cost_type='24hr')
    nhb_ext_out = ext.run(
        trip_origin='nhb',
        cost_type='24hr')

    dist = dm.DistributionModel(
        config_path,
        params_file)

    int_hb = dist.run(
        tlb_area='north',
        segmentation='tfn',
        distribution_segments=params['hb_distribution_segments'],
        dist_function='ln',
        trip_origin='hb',
        cost_type='24hr',
        furness_loops=1999,
        fitting_loops=100,
        iz_cost_infill=.5,
        export_modes=params['output_modes'],
        echo=True,
        mp_threads=-1)

    int_nhb = dist.run(
        tlb_area='north',
        segmentation='tfn',
        distribution_segments=params['nhb_distribution_segments'],
        dist_function='ln',
        trip_origin='nhb',
        cost_type='tp',
        furness_loops=1999,
        fitting_loops=100,
        iz_cost_infill=.5,
        export_modes=params['output_modes'],
        echo=True,
        mp_threads=-1)

    # Compile tp pa
    pa2od.build_tp_pa(file_drive=params['base_directory'],
                      model_name=params['model_name'],
                      iteration=params['iteration'],
                      distribution_segments=params['hb_output_segments'],
                      internal_input='synthetic',
                      external_input='synthetic',
                      write_modes=[3],
                      arrivals=False,
                      write=True)

    # *Compile tp pa @ 24hr
    f.build_tp_pa(file_drive=params['base_directory'],
                  model_name=params['model_name'],
                  iteration=params['iteration'],
                  distribution_segments=params['hb_output_segments'],
                  internal_input='synthetic',
                  external_input='synthetic',
                  write_modes=[3],
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

    # Import compilation params
    pa_compilation_params, od_compilation_params = nup.get_compilation_params(lookup_folder)

    if params['compile_pa']:
        compiled_pa_matrices = nup.compile_pa(o_paths['pa'],
                                              (i_paths['lookups'] +
                                               '/' +
                                               pa_compilation_params))  # Split time = true

        # Export compiled PA
        for mat in compiled_pa_matrices:
            for key, value in mat.items():
                print(key)
                c_pa_out = (o_paths['c_pa_export'] +
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
        vo.people_vehicle_conversion(input_folder=os.path.join(params['base_directory'],
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
    # ## PREPARE OUTPUTS ## #
    print("Initialising outputs...")
    write_input_info(
        os.path.join(params['base_directory'], "input_parameters.txt"),
        version.__version__,
        params['iteration'],
        params['model_name'],
        params['land_use_zoning']
    )


    def write_input_info(output_path: str,
                         tms_version: str,
                         model_name: str,
                         land_use_zoning: str,
                         ) -> None:

        out_lines = [
            'TMS version: ' + str(tms_version),
            'Model Name: ' + str(model_name),
            'Run Date: ' + str(time.strftime('%D').replace('/', '_')),
            'Start Time: ' + str(time.strftime('%T').replace('/', '_')),
            "Land Use Zoning System: " + str(land_use_zoning),

        ]
        with open(output_path, 'w') as out:
            out.write('\n'.join(out_lines))
