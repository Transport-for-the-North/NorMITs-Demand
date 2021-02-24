"""
Run the Travel Market Synthesiser
"""

import os

import pandas as pd

import normits_demand.models.tms as tms
import normits_demand.models.production_model as pm
import normits_demand.models.attraction_model as am
import normits_demand.models.external_model as em
import normits_demand.models.distribution_model as dm

def main(config_path = 'I:/NorMITs Synthesiser/config/'):

    """
    Wrapper function to run TMS start to finish based on specified params.
    """

    # Ask user which config file to use
    params = [x for x in os.listdir(config_path) if 'config' in x]
    if len(params) == 0:
        raise ValueError('no trip length bands in folder')
    for (i, option) in enumerate(params, 0):
        print(i, option)
        selection_c = input('Choose a config file (index): ')
        params_file = pd.read_csv(
            os.path.join(config_path,
                         params[int(selection_c)])
        )

    tms_run = tms.TMSPathing(config_path,
                             params_file)

    # Check status of lookup folder
    tms_run.lookups = tms_run.lookup_audit()

    # BACKLOG: Project status
    tms_run.project_status = tms_run.project_check()

    # BACKLOG: Do this stuff based on the project status

    p = pm.ProductionModel(config_path, params_file)
    p.ping_outpath()
    if p.run_dict['hb_p_run'] == '':
        hb_p_out = p.run_hb(verbose=True)

    a = am.AttractionModel(config_path, params_file)
    a.ping_outpath()
    if a.run_dict['hb_a_run'] == '':
        a_out = a.run()

    if p.run_dict['nhb_p']:
        nhb_p_out = p.run_nhb(
            production_vector=p.export['out_hb'],
            attraction_vector=a.export['out_hb'])

    # Delete trip end models
    del p, a
    # Update project status
    # BACKLOG: Project status

    # Run HB external model
    ext = em.ExternalModel(
        config_path,
        params_file)
    hb_ext_out = ext.run(
        trip_origin='hb',
        cost_type='24hr')  # Vectors??
    nhb_ext_out = ext.run(
        trip_origin='hb',
        cost_type = 'tp')

    """
    ext_hb = em.ExternalModel(file_drive=params['base_directory'],
                              model_name=params['model_name'],
                              iteration=params['iteration'],
                              tlb_area='gb',
                              segmentation='tfn',
                              distribution_segments=params['hb_distribution_segments'],
                              trip_origin='hb',
                              cost_type='24hr',
                              export_ext_modes=[],
                              export_non_dist_modes=['6'])

    # Run NHB external model
    ext_nhb = em.ExternalModel(file_drive=params['base_directory'],
                               model_name=params['model_name'],
                               iteration=params['iteration'],
                               tlb_area='gb',
                               segmentation='tfn',
                               distribution_segments=params['nhb_distribution_segments'],
                               trip_origin='nhb',
                               cost_type='tp',
                               export_ext_modes=[],
                               export_non_dist_modes=['6'])
    """

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

    # TODO: Run new segment split function
    import_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/PA Matrices'
    target_tld_folder = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/enhanced_segments'
    # TODO: Using single tld for whole country - run North and GB and compare
    base_hb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/hb_productions_noham.csv'
    base_nhb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/nhb_productions_noham.csv'
    base_hb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_hb_attractions.csv'
    base_nhb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_nhb_attractions.csv'
    export_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/Segmented Distributions'
    # lookup_folder = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'
    # replaced above lookup_folder with NoRMS model zone lookups path
    lookup_folder = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups'

    base_productions_path = base_hb_productions
    base_attractions_path = base_hb_attractions

    sd.disaggregate_segments(import_folder,
                             target_tld_folder,
                             base_productions_path,
                             base_attractions_path,
                             export_folder,
                             lookup_folder,
                             aggregate_surplus_segments=True,
                             rounding=5,
                             trip_origin='hb',
                             iz_infill=0.5,
                             furness_loops=1999,
                             mp_threads=-1)

    base_productions_path = base_nhb_productions
    base_attractions_path = base_nhb_attractions

    sd.disaggregate_segments(import_folder,
                             target_tld_folder,
                             base_productions_path,
                             base_attractions_path,
                             export_folder,
                             lookup_folder,
                             aggregate_surplus_segments=True,
                             rounding=5,
                             trip_origin='nhb',
                             tp='tp',
                             iz_infill=0.5,
                             furness_loops=1999,
                             mp_threads=-1)

    if not params['rail_fusion']:
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

    elif params['rail_fusion']:

        # Run fusion engine integration
        # TODO: Call rail fusion

        # TODO:
        # Filter init_params to rail only
        rail_hb_init_params = init_params.copy()
        rail_hb_init_params = rail_hb_init_params[rail_hb_init_params['m'] == 6]

        if pa2od.build_tp_pa(params['hb_output_segments'],
                             rail_hb_init_params,
                             model_name,
                             productions,
                             internal_import=o_paths['fusion_summaries'],
                             external_import=o_paths['external'],
                             pa_export='D:/NorMITs Synthesiser/',
                             write_modes=[6],
                             arrivals=False,
                             write=True):
            print('Transposed PA to tp PA')

        # Do nhb TP
        # 24hr OD to TP OD for NHB
        if pa2od.build_tp_pa(['p', 'm'],
                             nhb_rail_init_params,
                             model_name,
                             nhb_productions,
                             internal_import=o_paths['fusion_summaries'],
                             external_import=o_paths['external'],
                             pa_export='D:/NorMITs Synthesiser/',
                             write_modes=[6],
                             arrivals=False,
                             write=True):
            print('Transposed NHB PA to tp PA and OD')
        else:
            raise ValueError('TP PA failed')

        # Build fusion OD
        fusion_od_audit = pa2od.build_od(params['hb_output_segments'],
                                         model_name,
                                         rail_hb_init_params,
                                         input_folder=o_paths['fusion_pa_export'],
                                         output_folder='D:/NorMITs Synthesiser/OD',
                                         phi_type='fhp_tp',
                                         export_modes=[6],
                                         write=True)
        print(fusion_od_audit)

    # Do NHB manually, copy from PA, rename OD

    # Get compilation params

    # Compile fusion PA
    if pa_compilation_params is not None:
        compiled_pa_matrices = nup.compile_pa(o_paths['fusion_pa_export'],
                                              (i_paths['lookups'] +
                                               '/' +
                                               pa_compilation_params))  # Split time = true

        # Export compiled PA
        for mat in compiled_pa_matrices:
            for key, value in mat.items():
                print(key)
                c_pa_out = (o_paths['c_fusion_pa_export'] +
                            '/' +
                            key +
                            '.csv')
                value.to_csv(c_pa_out, index=False)

    # Compile fusion OD
    if od_compilation_params is not None:

        compiled_od_matrices = nup.compile_od(o_paths['fusion_od_export'],
                                              (i_paths['lookups'] +
                                               '/' +
                                               od_compilation_params))

        for mat in compiled_od_matrices:
            for key, value in mat.items():
                print(key)
                c_od_out = (o_paths['c_fusion_od_export'] +
                            '/' +
                            key +
                            '.csv')
                value = value.sort_values(['o_zone', 'd_zone']).reset_index(drop=True)
                value.to_csv(c_od_out, index=False)

    ra.distribution_report(params['base_directory'],
                           model_name=params['model_name'],
                           iteration=params['iteration'],
                           model_segments=['p', 'm', 'ca'],
                           distributions='Fusion Outputs/Fusion OD Matrices',
                           matrix_format='wide',
                           report_tp='tp',
                           internal_reports=True,
                           write=True)

    ra.distribution_report(params['base_directory'],
                           model_name=params['model_name'],
                           iteration=params['iteration'],
                           model_segments=['p', 'm', 'ca'],
                           distributions='Fusion Outputs/Compiled Fusion PA Matrices',
                           matrix_format='long',
                           report_tp='tp',
                           internal_reports=True,
                           write=True)

    ra.distribution_report(params['base_directory'],
                           model_name=params['model_name'],
                           iteration=params['iteration'],
                           model_segments=['p', 'm', 'ca'],
                           distributions='Fusion Outputs/Compiled Fusion OD Matrices',
                           matrix_format='long',
                           report_tp='tp',
                           internal_reports=True,
                           write=True)

    # TODO: Run od 2 pa from post-me mats

    # Run segment split function
    import_folder = 'Y:/NorMITs Demand/noham/v2_2-EFS_Output/iter1/Matrices/Post-ME Matrices/24hr PA Matrices'
    target_tld_folder = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/enhanced_segments'
    # TODO: Using single tld for whole country - run North and GB and compare
    base_hb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/hb_productions_noham.csv'
    base_nhb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/nhb_productions_noham.csv'
    base_hb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_hb_attractions.csv'
    base_nhb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_nhb_attractions.csv'
    export_folder = 'Y:/NorMITs Demand/noham/v2_2-EFS_Output/iter1/Matrices/Segmented Post-ME Matrices'
    # lookup_folder = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'
    # replaced above lookup_folder with NoRMS model zone lookups path
    lookup_folder = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups'

    base_productions_path = base_hb_productions
    base_attractions_path = base_hb_attractions

    sd.disaggregate_segments(import_folder,
                             target_tld_folder,
                             base_productions_path,
                             base_attractions_path,
                             export_folder,
                             lookup_folder,
                             aggregate_surplus_segments=True,
                             rounding=5,
                             trip_origin='hb',
                             iz_infill=0.5,
                             furness_loops=1999,
                             mp_threads=-1)

    base_productions_path = base_nhb_productions
    base_attractions_path = base_nhb_attractions

    sd.disaggregate_segments(import_folder,
                             target_tld_folder,
                             base_productions_path,
                             base_attractions_path,
                             export_folder,
                             lookup_folder,
                             aggregate_surplus_segments=True,
                             rounding=5,
                             trip_origin='nhb',
                             tp='tp',
                             iz_infill=0.5,
                             furness_loops=1999,
                             mp_threads=-1)
    """
