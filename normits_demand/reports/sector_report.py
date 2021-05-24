# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:34:00 2020

@author: genie
"""

import os

import pandas as pd
import numpy as np

from typing import List

import normits_demand as nd
import normits_demand.utils as nup
from normits_demand.utils import n_matrix_split as ms
from normits_demand.matrices import translate_matrices
import normits_demand.utils.file_ops as fo

class SectorReporter:
    """
    Class to build sector reports from an output demand vector
    """

    def __init__(self,
                 target_folder: nd.PathLike,
                 model_name: str,
                 output_folder: nd.PathLike,
                 input_type: str = None,
                 model_schema: nd.PathLike = None,
                 model_sectors: nd.PathLike = None,
                 target_file_types: List = ['.csv', '.bz2']
                 ):

        """
        target_folder: file or folder of files to do sector reports for
        model_name: name of the model to look for
        input_type: from 'vector', 'long_matrix', 'wide_matrix'
        model_schema: folder to look for correspondences. will search in
        default folder if nothing provided
        model_sectors = path to .csv correspondence to sectors
        """

        # Init
        _default_import_folder = 'I:/NorMITs Demand/import'

        self.target_folder = target_folder

        # Pass init params to object
        self.model_name = model_name
        self.zone_id = '%s_zone_id' % model_name.lower()

        # If no model schema folder - find one
        if model_schema is None:
            home_list = os.listdir(_default_import_folder)
            model_schema = [x for x in home_list if model_name in x][0]
            model_schema = os.path.join(
                _default_import_folder,
                model_schema,
                'model schema')

        # Figure out input type
        # TODO: If nothing provided, work it out
        self.input_type = input_type

        # Set model schema
        self.model_schema = model_schema

        # Find sector correspondence
        if model_sectors is None:
            schema_dir = os.listdir(model_schema)
            corr_name = '%s_sector_correspondence.csv' % model_name.lower()
            sector_file = [x for x in schema_dir if corr_name in x][0]
            model_sectors = os.path.join(model_schema, sector_file)

        self.sectors = pd.read_csv(model_sectors)

        self.target_file_types = target_file_types



    def sector_report(self,
                      sector_type: str = 'ca_sector',
                      ca_report: bool = True,
                      three_sector_report: bool = True,
                      export: bool = False):

        """
        sector_type: report requested - takes
            'ca_sector', 'three_sector', 'ie_sector'
        export = False:
            Write to object output dir, or not
        """

        # Index folder
        # TODO: Pull imports and parsing into line with NorMITs standard
        target_mats = os.listdir(self.target_folder)
        # Filter down to target file types
        target_mats = [x for y in self.target_file_types for x in target_mats if y in x]

        # Subset sectors into ie and 3 sector reports
        ca_sectors_2d = translate_matrices.convert_correspondence_to_wide(
            long_correspondence=self.sectors,
            primary_key=self.zone_id,
            secondary_key='ca_sector_2020_zone_id')
        three_sectors_2d = translate_matrices.convert_correspondence_to_wide(
            long_correspondence=self.sectors,
            primary_key=self.zone_id,
            secondary_key='three_sector_id')
        ie_2d = translate_matrices.convert_correspondence_to_wide(
            long_correspondence=self.sectors,
            primary_key=self.zone_id,
            secondary_key='ie_id')

        # Apply translation
        mat_sector_reports = dict()
        # TODO: Assumptions galore - needs to be smarter and better integrated
        for tm in target_mats:

            print(tm)
            mat = fo.read_df(os.path.join(self.target_folder, tm))

            mat = mat.rename(columns={list(mat)[0]: 'norms_zone_id'})

            if self.input_type == 'wide_matrix':

                """
                long_data = pd.melt(mat,
                                    id_vars=list(mat)[0],
                                    var_name='a_zone',
                                    value_name='demand',
                                    col_level=0)
                                    
                sector_report = 
                """



            elif self.input_type == 'long_matrix':
                # TODO: Translate to wide, do mat wide trans
                print('Can\'t do these yet')
            elif self.input_type == 'vector':
                # TODO: Do simple vector trans by required subset(s)
                print('Can\'t do these yet')



        # Export

        sector_report = ''

        return sector_report

    def _sectors_join_method(self,
                             long_data):

        """
        Method for joining sectors length wise
        Expects format 'p_zone', 'a_zone', 'demand'

        """

        long_data = long_data.merge(self.sectors,
                                    how='left',
                                    on=self.zone_id)

        long_data = long_data.reindex(
            ['ca_sector_2020_zone_id', 'a_zone', 'demand'], axis=1)
        long_data = long_data.groupby(['ca_sector_2020_zone_id', 'a_zone']).sum().reset_index()

        long_data = long_data.rename(columns={'ca_sector_2020_zone_id': 'sector_p',
                                              'a_zone': self.zone_id})
        long_data['norms_zone_id'] = long_data[self.zone_id].astype(int)

        left_only = long_data.copy()
        left_only_sum = left_only.reindex(
            ['sector_p', 'demand'], axis=1).groupby('sector_p').sum().reset_index()

        long_data = long_data.merge(self.sectors,
                                    how='left',
                                    on=self.zone_id)

        long_data = long_data.reindex(
            ['sector_p', 'ca_sector_2020_zone_id', 'demand'], axis=1)
        long_data = long_data.groupby(['sector_p', 'ca_sector_2020_zone_id']).sum().reset_index()

        long_data = long_data.rename(columns={'ca_sector_2020_zone_id': 'sector_a'})

        pivoted_data = pd.pivot(long_data, index='sector_p', columns='sector_a', values='demand')

        return long_data, pivoted_data

    def _matrix_zone_translation(self,
                                 mat,
                                 sectors_mat):

        """
        mat = dataframe of wide matrix, index in first column
        """
        mat = mat.set_index(list(mat)[0])

        np_mat = mat.to_numpy()
        # TODO: do 2d vector trans

        sectors_mat = sectors_mat.values

        zoning_len = sectors_mat.shape[1]
        sector_len = sectors_mat.shape[-1]

        sectors_cube = np.broadcast_to(
            sectors_mat, (zoning_len,
                          zoning_len,
                          sector_len))

        row_cube = np.zeros(sectors_cube.shape)
        col_cube = np.zeros(sectors_cube.shape)
        for i in range(sector_len):
            multiplier = sectors_cube[:, :, i]
            # Don't think I need both except to arrive at row totals
            row_slice = np_mat * multiplier
            row_cube[:, :, i] = row_slice
            col_slice = np_mat.T * multiplier
            col_cube[:, :, i] = col_slice

        row_sum = row_cube.sum(axis=1)  # from
        col_sum = col_cube.sum(axis=0)  # to

        for i in range(zoning_len):
            print(i)
            sectors_mat
            row_slice = row_sum[i]
            col_slice = col_sum[i]

        row_totals = row_sum.sum(axis=0)
        col_totals = col_sum.sum(axis=0)

"""

for mat in pcu_mats:
    print(mat)
    dat = pd.read_csv(os.path.join(od_vehicles,
                                   mat), header=header)
    dat_cols = list(dat)
    
    # TODO: test input format on list length

    if import_format == 'long':
        # Make it wide again
        
        dat = dat.pivot(index=dat_cols[0], columns=dat_cols[1], values=dat_cols[2]).values
        
        audit_in = dat.sum()
        
        report = ms.n_matrix_split(dat,
                                    indices=[north_sectors, scotland_sectors, south_sectors],
                                    index_names=['1', '2', '3'],
                                    summarise=True)

    elif import_format == 'wide':
        dat = dat.drop(list(dat)[0]).values

        report = ms.n_matrix_split(dat,
                                    indices=[north_sectors,
                                             scotland_sectors,
                                             south_sectors],
                                    index_names=['1', '2', '3'],
                                    summarise=True)

    row_frame = pd.DataFrame.from_dict(report)
    
    audit_out = row_frame['dat'].sum()

    if round(audit_in, 3) == round(audit_out, 3):
        print('Audit in same as audit out')
    else:
        raise Warning('Report total different from in values')

    cols = row_frame['name'].str.split('_to_', expand=True)
    cols = cols.rename(columns={0: 'from',
                                1: 'to'})
    row_frame['from'] = cols['from']
    row_frame['to'] = cols['to']
    row_frame=row_frame.drop('name', axis=1).reindex(['from', 'to', 'dat'], axis=1)

    row_frame = row_frame.pivot(index='from', columns='to', values='dat')

    out_name = mat.replace('od', '3_sector_report_od')

    row_frame.to_csv(os.path.join(export, out_name), index=False)
"""