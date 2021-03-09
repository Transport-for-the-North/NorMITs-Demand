# -*- coding: utf-8 -*-
"""
Attraction Model for NorMITs Demand
"""
# Modules required
import os # File operations

import pandas as pd
import numpy as np

import normits_demand.build.pathing as demand

import normits_demand.trip_end_constants as tec

from normits_demand.utils import utils as nup
from normits_demand.utils import compress as com
from normits_demand.constraints import ntem_control as ntem
from normits_demand.utils.general import safe_dataframe_to_csv


class AttractionModel(demand.Pathing):

    """
    NorMITs Attraction model.
    Attraction end of NorMITs Trip End Model in the NorMITs Demand suite.
    """

    def __init__(self,
                 config_path,
                 param_file
                 ):
        super().__init__(config_path,
                         param_file)

        # Check for exports
        self.export = self.ping_outpath()

    def get_attractions(self
                        ):
        """
        Imports raw attraction data from either a flat file on the the Y://
        or direct from the SQL server.

        Parameters
        ----------
        source:
            Where to look for the data. Will take 'flat' or 'sql', now just
            flat.

        flat_source:
            Path to flat source. Just points at global import path.

        Returns
        ----------
        all_msoa:
            All attractions by modelling employment types at msoa level.
        """
        flat_source = os.path.join(self.import_folder,
                                   'attraction_data',
                                   self.params['non_resi_land_use_path'])

        # Blocking out most of this so it just imports everything.
        # BACKLOG: Adapt to handle the other 2 required datasets. Process if needed.
        all_msoa = pd.read_csv(flat_source)

        return all_msoa

    def _profile_attractions(self,
                             msoa_attractions,
                             attr_profile,
                             soc_zero=True):
        """
        Takes an attraction profile defined by an import file in a model folder
        and creates attractions to those segments.

        Parameters
        ----------
        model_folder:
            Takes a model folder to look for an internal area definition.

        msoa_attractions:
            DataFrame containing msoa attractions by E segments, as imported from
            HSL data.

        Returns
        ----------
        a_list:
            A list of dataframe for each attraction type.
        """

        # Build all job list for iteration
        all_jobs = list(msoa_attractions)
        all_jobs.remove('msoa_zone_id')

        # Pivot attraction list
        attr_profile = attr_profile.melt(
            id_vars=['purpose', 'ntem_purpose'],
            var_name='employment_cat',
            value_name='a_factor'
        )

        attractions = msoa_attractions.merge(
            attr_profile,
            how='left',
            on=['employment_cat']
        )

        attractions['2018'] *= attractions['a_factor']

        attractions = attractions.reindex(
            ['msoa_zone_id', 'ntem_purpose', 'soc', '2018'],
            axis=1).groupby(['msoa_zone_id', 'ntem_purpose', 'soc']).sum(
        ).reset_index()

        # TODO: Just import with p in
        attractions = attractions.rename(columns={'ntem_purpose': 'p'})

        if soc_zero:
            attractions_s0 = attractions.groupby(
                ['msoa_zone_id', 'p']).sum().reset_index()
            attractions_s0['soc'] = 0

            attractions = pd.concat([attractions, attractions_s0])
            attractions = attractions.reindex(
                ['msoa_zone_id', 'p', 'soc', '2018'],
                axis=1).sort_values(
                ['msoa_zone_id', 'p', 'soc']).reset_index(drop=True)

        return attractions

    def ping_outpath(self):

        """
        """
        # BACKLOG: This solves a lot of problems, integrate into main
        output_dir = os.path.join(self.run_folder,
                                  self.params['iteration'])
        output_f = 'Attraction Outputs'

        in_hb = os.path.join(
            output_dir,
            output_f,
            'hb_attractions_' +
            self.params['land_use_zoning'].lower() +
            '.csv')

        in_nhb = os.path.join(
            output_dir,
            output_f,
            'hb_attractions_' +
            self.params['land_use_zoning'].lower() +
            '.csv')

        out_hb = os.path.join(
            output_dir,
            output_f,
            'hb_attractions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        out_nhb = os.path.join(
            output_dir,
            output_f,
            'nhb_attractions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        if not os.path.exists(in_hb):
            in_hb = ''
        if not os.path.exists(in_nhb):
            in_nhb = ''
        if not os.path.exists(out_hb):
            out_hb = ''
        if not os.path.exists(out_nhb):
            out_nhb = ''

        export_dict = {'in_hb': in_hb,
                       'in_nhb': in_nhb,
                       'out_hb': out_hb,
                       'out_nhb': out_nhb}

        self.export = export_dict

        return export_dict

    def _get_attraction_weights(self,
                                trip_origin: str = 'hb'):

        """
        :param trip_origin:
        string of 'hb' or 'nhb'
        :return:
        """
        if trip_origin not in ['hb', 'nhb']:
            raise ValueError('Invalid trip origin')

        aw_path = os.path.join(
            self.import_folder,
            'attraction_params',
            self.params[trip_origin + '_attraction_weights']
        )

        att_w = pd.read_csv(aw_path)

        return att_w

    def _get_mode_split(self,
                        trip_origin='hb'):
        """:arg
        """

        mode_split = pd.read_csv(
            os.path.join(self.import_folder,
                         'attraction_params',
                         trip_origin +
                         '_' +
                         self.params['attraction_mode_split']))
        return mode_split

    @staticmethod
    def _apply_mode_splits(attractions,
                           mode_splits):
        """
        """
        # TODO: Relativise group and sum cols

        attractions = attractions.merge(
            mode_splits,
            how='left',
            on=['msoa_zone_id', 'p'])

        attractions['2018'] *= attractions['mode_share']
        attractions = attractions.drop('mode_share', axis=1)

        attractions = attractions.reindex(
            ['msoa_zone_id', 'p', 'm', 'soc', '2018'],
            axis=1).groupby(['msoa_zone_id', 'p', 'm', 'soc']).sum(
        ).reset_index()

        attractions = attractions.sort_values(
            ['msoa_zone_id', 'p', 'm', 'soc']).reset_index(drop=True)

        return attractions

    def _balance_a_to_p(attractions,
                        productions,
                        attr_var_col='2018',
                        prod_var_col='trips',
                        zone_col='msoa_zone_id'):
        """
        """

        # Get unq p segments
        unq_seg = list(productions)
        unq_seg.remove(prod_var_col)
        unq_seg.remove(zone_col)
        # TODO: Please Chris, learn to use itertools
        us_iter = productions.reindex(
            unq_seg,
            axis=1).drop_duplicates().reset_index(drop=True)

        for i, row in us_iter.iterrows():
            print(row)




        return attractions

    def run(self,
            trip_origin='hb',
            control_to_productions=False,
            productions_path=None
            ):

        """
        Function to run the attraction model. Takes a path to a lookup folder
        containing

        Returns
        ----------
        attractions:
            List of DataFrames containing attractions, split by attraction type.
            At ONS geography provided at input.

        all_attr:
            Single DataFrame with all attraction splits side by side. At ONS
            geography provided at input.

        zonal_attractions:
            List of DataFrames containing attractions, split by attraction type.
            At target model zoning system.

        all_zonal_attr:
            Single DataFrame with all attraction splits side by side.
            At target model zoning system.

        """
        output_dir = os.path.join(self.run_folder,
                                  self.params['iteration'],
                                  'Attraction Outputs')
        nup.create_folder(output_dir, chDir=False)

        print("Getting MSOA attractions")
        # Filters down to internal only here.
        # Uses global variables that aren't in the function call & shouldn't.
        msoa_attractions = self.get_attractions()

        all_jobs = list(msoa_attractions)
        all_jobs.remove('msoa_zone_id')

        # Import attraction profile
        attr_profile = self._get_attraction_weights(trip_origin=trip_origin)

        # Profile and weight attractions
        attractions = self._profile_attractions(
            msoa_attractions, attr_profile, soc_zero=True)

        # Attach msoa mode_splits
        mode_splits = self._get_mode_split(trip_origin=trip_origin)

        attractions = self._apply_mode_splits(attractions,
                                              mode_splits)

        ntem_lad_lookup = pd.read_csv(tec.MSOA_LAD)

        if self.params['attraction_ntem_control']:
            # Do an NTEM adjustment
            ntem_totals = pd.read_csv(self.params['ntem_control_path'])

            attractions, adj, audit, lad = ntem.control_to_ntem(
                attractions,
                ntem_totals,
                ntem_lad_lookup,
                base_value_name='2018',
                ntem_value_name='Attractions',
                trip_origin='hb')
            print(audit)

            if self.params['export_lad']:
                lad.to_csv(
                    os.path.join(output_dir,
                                 'lad_' +
                                 trip_origin +
                                 '_attractions.csv'),
                    index=False)

        if self.params['export_uncorrected']:
            # TODO: Export compressed
            safe_dataframe_to_csv(
                attractions,
                os.path.join(
                    output_dir,
                    trip_origin +
                    '_attractions_uncorrected.csv'),
                index=False)

        # Control to k factors
        if self.params['attraction_k_factor_control']:
            print('...')
            # Adjust to k factor for hb
            # BACKLOG: reliable k-factor adjustment with fixed input

        if control_to_productions:
            productions = com.read_in(productions_path)
            
            attractions = _balance_a_to_p(attractions,
                                          productions)


            nup.balance_a_to_p(ia_name,
                           productions,
                           attractions,
                           p_var_name='productions',
                           a_var_name='attractions',
                           round_val=None,
                           echo=True)


        # Write input attractions
        if self.params['export_msoa']:
            attractions.to_csv(
                os.path.join(
                    output_dir,
                    'hb_attractions' +
                    self.params['land_use_zoning'].lower() +
                    '.csv'),
                index=False)

        # Aggregate input productions to model zones - not yet
        """
        if control_to_productions:
            productions_path=None
        
        zonal_hb_attr = self.aggregate_to_model_zones_attr(
            hb_attr,
            model_zone_lookup_path,
            translation_name='overlap_msoa_split_factor',
            max_level=True)

        zonal_nhb_attr = self.aggregate_to_model_zones_attr(
            nhb_attr,
            model_zone_lookup_path,
            translation_name='overlap_msoa_split_factor',
            max_level=True)

        hb_out_path = os.path.join(
            output_dir,
            'hb_attractions_' +
            self.params['model_zoning'].lower() +
            '.csv')
        nhb_out_path = os.path.join(
            output_dir,
            'nhb_attractions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        # Write output totals
        if self.params['export_model_zoning']:
            zonal_hb_attr.to_csv(
                hb_out_path,
                index=False)
            zonal_nhb_attr.to_csv(
                nhb_out_path,
                index=False)
        """

        return attractions
