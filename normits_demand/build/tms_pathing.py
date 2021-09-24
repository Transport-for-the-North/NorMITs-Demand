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
        model_path = os.path.join(self.run_folder, self.params['iteration'])

        # Set production path
        production_path = os.path.join(model_path, 'Production Outputs')

        # Set attraction path
        attraction_path = os.path.join(model_path, 'Attraction Outputs')

        # Create project folders
        distribution_path = os.path.join(model_path, 'Distribution Outputs')
        nup.create_folder(distribution_path, chDir=False)

        reports = os.path.join(distribution_path, 'Logs & Reports')
        nup.create_folder(reports, chDir=False)

        external_export = os.path.join(distribution_path, 'External Distributions')
        nup.create_folder(external_export, chDir=False)

        non_dist_out = os.path.join(distribution_path, '24hr Non Dist Matrices')
        nup.create_folder(non_dist_out, chDir=False)

        # Compile into import and export
        self.tms_in = {
            'hb_p': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\hb_productions\hb_msoa_notem_segmented_2018_dvec.pkl",
            'hb_a': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\hb_attractions\hb_msoa_notem_segmented_2018_dvec.pkl",
            'nhb_p': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\nhb_productions\nhb_msoa_notem_segmented_2018_dvec.pkl",
            'nhb_a': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\nhb_attractions\nhb_msoa_notem_segmented_2018_dvec.pkl"}

        self.tms_out = {
            'p': production_path,
            'a': attraction_path,
            'external': external_export,
            'non_dist': non_dist_out,
            'reports': reports}

        # BACKLOG: Check for init params or else make them

    # BACKLOG: PA to OD handling here?
