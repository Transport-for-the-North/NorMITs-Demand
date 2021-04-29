"""
Run the Travel Market Synthesiser
"""

import os
import pandas as pd
import importlib as ri

import normits_demand.build.tms_pathing as tms
import normits_demand.build.project as prj
import normits_demand.models.production_model as pm
import normits_demand.models.attraction_model as am
import normits_demand.models.external_model as em
import normits_demand.models.distribution_model as dm


if __name__ == '__main__':

    config_path = 'I:/NorMITs Synthesiser/config/'

    # Ask user which config file to use
    params_file = prj.select_params_file(config_path)

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
                    control_to_productions = True,
                    productions_path = p.export['in_nhb'])

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
        cost_type = '24hr')

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
    pa2od.build_tp_pa(file_drive=params['base_directory'],
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
