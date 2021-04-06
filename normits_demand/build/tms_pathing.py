# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:43:09 2020
"""
import os

import normits_demand.build.pathing as pathing
import normits_demand.utils.utils as nup

class TMSPathing(pathing.Pathing):

    def __init__(
            self,
            config_path,
            param_file):
        super().__init__(config_path,
                         param_file)

        # Set top level model folder
        model_path = os.path.join(
            self.run_folder,
            self.params['iteration'])

        # Set production path
        production_path = os.path.join(
            model_path,
            'Production Outputs')

        # Set attraction path
        attraction_path = os.path.join(
            model_path,
            'Attraction Outputs')

        # Production import path
        hb_p_import_path = os.path.join(
                production_path,
                'hb_productions_' +
                self.params['model_zoning'].lower() +
                '.csv')
        hb_a_import_path = os.path.join(
                attraction_path,
                'hb_attractions_' +
                self.params['model_zoning'].lower() +
                '.csv')

        nhb_p_import_path = os.path.join(
                production_path,
                'nhb_productions_' +
                self.params['model_zoning'].lower() +
                '.csv')
        nhb_a_import_path = os.path.join(
                attraction_path,
                'nhb_attractions_' +
                self.params['model_zoning'].lower() +
                '.csv')

        # Create project folders
        distribution_path = os.path.join(
            model_path,
            'Distribution Outputs')
        nup.create_folder(distribution_path, chDir=False)

        reports = os.path.join(
            distribution_path,
            'Logs & Reports')
        nup.create_folder(reports, chDir=False)

        external_export = os.path.join(
            distribution_path,
            '/External Distributions')
        nup.create_folder(external_export, chDir=False)

        non_dist_out = os.path.join(
            distribution_path,
            '/24hr Non Dist Matrices')
        nup.create_folder(non_dist_out, chDir=False)

        # Compile into import and export
        self.tms_in = {
            'hb_p': hb_p_import_path,
            'hb_a': hb_a_import_path,
            'nhb_p': nhb_p_import_path,
            'nhb_a': nhb_a_import_path}

        self.tms_out = {
            'p': production_path,
            'a': attraction_path,
            'external': external_export,
            'non_dist': non_dist_out,
            'reports': reports}

        # BACKLOG: Check for init params or else make them

    # BACKLOG: PA to OD handling here?
    """
    Sets paths for imports to be set as variables.
    Creates project folders.

    Parameters
    ----------
    file_drive = 'Y:/':
        Name of root drive to do work on. Defaults to TfN Y drive.

    model_name:
        Name of model as string. Should be same as model descriptions.

    iteration:
        Current iteration of model. Defaults to global default.

    Returns
    ----------
    [0] imports:
        Paths to all Synthesiser import parameters.

    [1] exports:
        Paths to all Synthesiser output parameters

    # Set base dir
    home_path = os.path.join(file_drive, 'NorMITs Synthesiser')

    # Set synth import folder
    import_path = os.path.join(home_path, 'import')

    # Set top level model folder, leave the slash on
    model_path = os.path.join(home_path,
                              model_name,
                              iteration)
    model_path += os.path.sep

    # Set model lookups location
    model_lookup_path = os.path.join(home_path,
                                     model_name,
                                     'Model Zone Lookups')

    # Set production path, leave slash on
    production_path = os.path.join(model_path, 'Production Outputs')
    production_path += os.path.sep

    # Set production path
    production_path = (model_path +
                       'Production Outputs/')

    # Set attraction path
    attraction_path = (model_path +
                     'Attraction Outputs/')

    # Production import path
    if trip_origin =='hb':
        p_import_path = (production_path +
                         model_name.lower() +
                         '_hb_internal_productions.csv')
        a_import_path = (attraction_path +
                         model_name.lower() +
                         '_hb_internal_attractions.csv')
    elif trip_origin == 'nhb':
        p_import_path = (production_path +
                         model_name.lower() +
                         '_nhb_internal_productions.csv')
        a_import_path = (attraction_path +
                         model_name.lower() +
                         '_nhb_internal_attractions.csv')
    # Raise user warning if no productions by this name
    if not os.path.exists(p_import_path):
        warnings.warn('No productions in folder.' +
                      'Check path or run production model')

    # Raise user warning if no productions by this name
    if not os.path.exists(a_import_path):
        warnings.warn('No attractions in folder.' +
                      'Check path or run attraction model')

    # Create project folders
    distribution_path = os.path.join(model_path, 'Distribution Outputs')
    nup.create_folder(distribution_path, chDir=False)

    fusion_path = os.path.join(model_path, 'Fusion Outputs')
    nup.create_folder(fusion_path, chDir=False)

    pcu_path = os.path.join(model_path, 'PCU Outputs')
    nup.create_folder(pcu_path)

    # Set distribution outputs (synthetic)
    reports = os.path.join(distribution_path, 'Logs & Reports')
    nup.create_folder(reports)

    summary_matrix_export = os.path.join(distribution_path, '24hr PA Distributions')
    nup.create_folder(summary_matrix_export)

    cjtw_hb_export = os.path.join(distribution_path, 'Cjtw PA Distributions')
    nup.create_folder(cjtw_hb_export)

    external_export = os.path.join(distribution_path, 'External Distributions')
    nup.create_folder(external_export)

    bin_export = os.path.join(distribution_path, 'Trip Length Distributions')
    nup.create_folder(bin_export)

    pa_export = os.path.join(distribution_path, 'PA Matrices')
    nup.create_folder(pa_export)

    pa_export_24 = os.path.join(distribution_path, 'PA Matrices 24hr')
    nup.create_folder(pa_export_24)

    arrival_export = os.path.join(distribution_path, 'D Arrivals')
    nup.create_folder(arrival_export)

    od_export = os.path.join(distribution_path, 'OD Matrices')
    nup.create_folder(od_export)

    me_export = os.path.join(distribution_path, 'PostME OD Matrices')
    nup.create_folder(me_export)

    compiled_pa_export = os.path.join(distribution_path, 'Compiled PA Matrices')
    nup.create_folder(compiled_pa_export)

    compiled_od_export = os.path.join(distribution_path, 'Compiled OD Matrices')
    nup.create_folder(compiled_od_export)

    # Set fusion exports
    fusion_summary_export = os.path.join(fusion_path, '24hr Fusion PA Distributions')
    nup.create_folder(fusion_summary_export)

    fusion_pa_export = os.path.join(fusion_path, 'Fusion PA Matrices')
    nup.create_folder(fusion_pa_export)

    fusion_od_export = os.path.join(fusion_path, 'Fusion OD Matrices')
    nup.create_folder(fusion_od_export)

    pcu_od_export = os.path.join(pcu_path, 'PCU OD Matrices')
    nup.create_folder(pcu_od_export)

    compiled_fusion_pa_export = os.path.join(fusion_path, 'Compiled Fusion PA Matrices')
    nup.create_folder(compiled_fusion_pa_export)

    compiled_fusion_od_export = os.path.join(fusion_path, 'Compiled Fusion OD Matrices')
    nup.create_folder(compiled_fusion_od_export)

    # Compile into import and export
    imports = {'imports': import_path,
               'lookups': model_lookup_path,
               'production_import': p_import_path,
               'attraction_import': a_import_path}

    exports = {'production_export': production_path,
               'attraction_export': attraction_path,
               'reports': reports,
               'summaries': summary_matrix_export,
               'cjtw': cjtw_hb_export,
               'external': external_export,
               'tld': bin_export,
               'pa': pa_export,
               'pa_24': pa_export_24,
               'od_export': od_export,
               'arrival_export': arrival_export,
               'me_export': me_export,
               'c_pa_export': compiled_pa_export,
               'c_od_export': compiled_od_export,
               'fusion_summaries': fusion_summary_export,
               'fusion_pa_export': fusion_pa_export,
               'fusion_od_export': fusion_od_export,
               'pcu_od_export': pcu_od_export,
               'c_fusion_pa_export': compiled_fusion_pa_export,
               'c_fusion_od_export': compiled_fusion_od_export}

    """
