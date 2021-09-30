# -*- coding: utf-8 -*-
"""
Created on: Wednesday Sept 22nd 2021

Original author: Nirmal Kumar

File purpose:
Tram inclusion on NoTEM outputs
"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os
import warnings

from typing import Dict, Tuple
from typing import List

# Third party imports
import pandas as pd
import numpy as np

# local imports
import normits_demand as nd

from normits_demand import efs_constants as consts

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils


class TramInclusion:
    """
    Tram Inclusion on NoTEM outputs
    """
    # Constants
    _return_segmentation_name = 'notem_hb_output'

    # Define wanted columns
    _target_col_dtypes = {
        'tram': {
            'msoa_zone_id': str,
            'p': int,
            'ca': int,
            'm': int,
            'trips': float
        },
        'notem': {
            'msoa_zone_id': str,
            'p': int,
            'ca': int,
            'm': int,
            'val': float
        },
        'notem_north': {
            'ie_sector_zone_id': int,
            'p': int,
            'ca': int,
            'm': int,
            'val': float
        }
    }

    def __init__(self,
                 tram_data_paths: nd.PathLike,
                 notem_outputs: Dict[str, nd.PathLike],
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """

        Parameters
        ----------
        tram_data_paths:
            The path to the tram data.


        """
        # Check that the paths we need exist!
        file_ops.check_file_exists(tram_data_paths)
        [file_ops.check_file_exists(x) for x in notem_outputs.values()]

        # Assign
        self.tram_data_paths = tram_data_paths
        self.notem_outputs = notem_outputs

    def run(self,
            verbose: bool = False) -> None:
        """
        Runs the tram inclusion part.
        """
        # Initialise timing
        # TODO(BT): Properly integrate logging
        start_time = timing.current_milli_time()
        du.print_w_toggle(
            "Starting Tram inclusion at: %s" % timing.get_datetime(),
            verbose=verbose
        )

        # Generate the productions for each year
        # for year in self.years:
        year_start_time = timing.current_milli_time()

        # ## GENERATE PURE DEMAND ## #
        du.print_w_toggle("Loading the tram data...", verbose=verbose)
        tram_data, notem_tram_seg = self._read_tram_data(verbose)
        tram_msoa, tram_zone_wo_infill = self._tram_infill_msoa(notem_tram_seg, tram_data, verbose)
        north_msoa, north_wo_infill = self._tram_infill_north(notem_tram_seg, tram_msoa, verbose)
        abc = self._non_tram_infill(tram_zone_wo_infill, north_wo_infill, tram_msoa, north_msoa, tram_data,
                                    notem_tram_seg)

    def _read_tram_data(self,
                        verbose: bool,
                        ) -> Tuple[pd.DataFrame, nd.DVector]:
        """
        Reads in the tram data.

        Parameters
        ----------
        verbose:
            Passed into the Dvector.

        Returns
        -------
        tram_dvec:
            Returns the tram Dvector
        """
        # Define the zoning and segmentations we want to use
        tram_zoning = nd.get_zoning_system('tram')
        tram_seg = nd.get_segmentation_level('hb_p_m7_ca')

        # Read the tram data
        tram_data = file_ops.read_df(
            path=self.tram_data_paths,
            find_similar=True,
        )
        cols = list(tram_data)
        cols.insert(2, 'm')
        tram_data['m'] = 7

        for col, dt in self._target_col_dtypes['tram'].items():
            tram_data[col] = tram_data[col].astype(dt)

        tram_data = tram_data.reindex(cols, axis=1)
        tram_data.rename(columns={'trips': 'val'}, inplace=True)
        notem_output_dvec = nd.from_pickle(self.notem_outputs['hb_p'])
        notem_tram_seg = notem_output_dvec.aggregate(out_segmentation=nd.get_segmentation_level('hb_p_m_ca'))

        return tram_data, notem_tram_seg

    def _tram_infill_msoa(self, notem_tram_seg, tram_data, verbose):

        notem_df = notem_tram_seg.to_df()

        for col, dt in self._target_col_dtypes['notem'].items():
            notem_df[col] = notem_df[col].astype(dt)
        # tram = nd.read_compressed_dvector(r"C:\Users\Godzilla\Documents\GitHub\NorMITs-Demand\normits_demand\core\definitions\zoning_systems\tram\zones.pbz2")

        notem_df = notem_df.loc[notem_df['msoa_zone_id'].isin(tram_data['msoa_zone_id'])]
        tram_zone_wo_infill = notem_df.copy()
        notem_df = notem_df.append(tram_data)

        notem_df['c_uniq_id'] = notem_df['msoa_zone_id'].astype(str) + '_' + notem_df['p'].astype(str) + '_' + \
                                notem_df['ca'].astype(str)
        notem_df['uniq_id'] = notem_df['msoa_zone_id'].astype(str) + '_' + notem_df['p'].astype(str) + '_' + \
                              notem_df['m'].astype(str) + '_' + notem_df['ca'].astype(str)
        notem_df.sort_values(by='msoa_zone_id')
        print(notem_df)
        print(notem_df.dtypes)
        notem_train = notem_df.loc[notem_df['m'] == 6]
        notem_train = notem_train.sort_values(by='c_uniq_id')
        notem_train.reset_index(inplace=True)

        # notem_train.reset_index(inplace=True)
        notem_tram = notem_df.loc[notem_df['m'] == 7]
        notem_tram = notem_tram.sort_values(by='c_uniq_id')
        notem_tram.reset_index(inplace=True)
        print(notem_train)
        print(notem_tram)

        notem_train['new_value'] = np.where(
            (notem_train['c_uniq_id'].isin(notem_tram['c_uniq_id'])) & (notem_train['val'] > notem_tram['val']),
            (notem_train['val'] - notem_tram['val']), 0)

        notem_tram['new_value'] = np.where(
            (notem_train['c_uniq_id'].isin(notem_tram['c_uniq_id'])) & (notem_train['val'] > notem_tram['val']),
            notem_tram['val'], notem_train['val'])
        print(notem_df)
        pivot = pd.pivot_table(notem_df, values='val', index=['m'], aggfunc='count')
        print(pivot)
        notem_df = notem_df[notem_df.m < 6]
        print(notem_df)
        notem_df['new_value'] = notem_df['val']
        # notem_train['val']=notem_train['new_value']
        # notem_train.drop(columns='new_value',inplace=True)
        # notem_tram['val'] = notem_tram['new_value']
        # notem_tram.drop(columns='new_value', inplace=True)
        print(notem_df)
        # print(notem_train)
        # notem_df.loc[notem_df['uniq_id'].isin(notem_train['uniq_id']), 'val'] = notem_train['new_value']
        notem_df = notem_df.append([notem_train, notem_tram])
        # notem_df = notem_df.append(notem_tram)
        notem_df.drop(columns='index', inplace=True)
        print(notem_df)

        notem_df = notem_df.sort_values(by=['msoa_zone_id', 'p', 'ca', 'm'])
        notem_df.to_csv(r"I:\Data\Light Rail\notem_df.csv", index=False)
        print(notem_df)
        uniq_id = notem_df['c_uniq_id'].drop_duplicates()
        notem_new_df = pd.DataFrame()
        # uniq_id = 'E02001023_1_1'
        for id in uniq_id:
            new_df = notem_df.loc[notem_df['c_uniq_id'] == id]

            sum_trips = new_df.loc[new_df['m'] < 7, 'val'].sum() - \
                        new_df.loc[new_df['m'] == 1, 'new_value'].sum() - \
                        new_df.loc[new_df['m'] == 2, 'new_value'].sum() - \
                        new_df.loc[new_df['m'] == 7, 'new_value'].sum()
            veh_trips = new_df.loc[(new_df['m'] > 2) & (new_df['m'] < 7), 'val'].sum()
            # new_df['final_val'] = new_df['new_value']
            new_df['final_val'] = np.where((new_df['m'] > 2) & (new_df['m'] < 7),
                                           ((new_df['val'] / veh_trips) * sum_trips), new_df['new_value'])

            notem_new_df = notem_new_df.append(new_df)
        print(notem_new_df)
        notem_new_df.to_csv(r"I:\Data\Light Rail\notem_new_df.csv", index=False)
        return notem_new_df, tram_zone_wo_infill

    def _tram_infill_north(self, notem_tram_seg, tram_msoa, verbose):

        ie_sectors = nd.get_zoning_system('ie_sector')
        notem_df = notem_tram_seg.translate_zoning(ie_sectors).to_df()
        print(notem_df)

        for col, dt in self._target_col_dtypes['notem_north'].items():
            notem_df[col] = notem_df[col].astype(dt)
        # tram = nd.read_compressed_dvector(r"C:\Users\Godzilla\Documents\GitHub\NorMITs-Demand\normits_demand\core\definitions\zoning_systems\tram\zones.pbz2")

        notem_df = notem_df.loc[notem_df['ie_sector_zone_id'] == 1]
        notem_df = notem_df.drop(['ie_sector_zone_id'], axis=1)
        north_wo_infill = notem_df.copy()
        print(notem_df)
        tram_data = tram_msoa[tram_msoa.m == 7]

        df_index_cols = ['p', 'm', 'ca', 'final_val']

        df_group_cols = df_index_cols.copy()
        df_group_cols.remove('final_val')
        tram_data = tram_data.reindex(df_index_cols, axis=1).groupby(df_group_cols).sum().reset_index()
        tram_data.rename(columns={'final_val': 'val'}, inplace=True)
        print(tram_data)
        notem_df = notem_df.append(tram_data).reset_index()
        print(notem_df)

        notem_df['c_uniq_id'] = notem_df['p'].astype(str) + '_' + \
                                notem_df['ca'].astype(str)

        notem_train = notem_df.loc[notem_df['m'] == 6]
        notem_train = notem_train.sort_values(by='c_uniq_id')
        notem_train.reset_index(inplace=True)

        # notem_train.reset_index(inplace=True)
        notem_tram = notem_df.loc[notem_df['m'] == 7]
        notem_tram = notem_tram.sort_values(by='c_uniq_id')
        notem_tram.reset_index(inplace=True)
        print(notem_train)
        print(notem_tram)

        notem_train['new_value'] = np.where(
            (notem_train['c_uniq_id'].isin(notem_tram['c_uniq_id'])) & (notem_train['val'] > notem_tram['val']),
            (notem_train['val'] - notem_tram['val']), 0)

        notem_tram['new_value'] = np.where(
            (notem_train['c_uniq_id'].isin(notem_tram['c_uniq_id'])) & (notem_train['val'] > notem_tram['val']),
            notem_tram['val'], notem_train['val'])
        print(notem_df)
        notem_df = notem_df[notem_df.m < 6]
        print(notem_df)
        notem_df['new_value'] = notem_df['val']

        print(notem_df)
        # print(notem_train)
        # notem_df.loc[notem_df['uniq_id'].isin(notem_train['uniq_id']), 'val'] = notem_train['new_value']
        notem_df = notem_df.append([notem_train, notem_tram])
        # notem_df = notem_df.append(notem_tram)
        notem_df.drop(columns='index', inplace=True)
        print(notem_df)

        notem_df = notem_df.sort_values(by=['p', 'ca', 'm'])
        notem_df.to_csv(r"I:\Data\Light Rail\notem_msoa_df.csv", index=False)
        print(notem_df)
        uniq_id = notem_df['c_uniq_id'].drop_duplicates()
        notem_new_df = pd.DataFrame()
        # uniq_id = 'E02001023_1_1'
        for id in uniq_id:
            new_df = notem_df.loc[notem_df['c_uniq_id'] == id]

            sum_trips = new_df.loc[new_df['m'] < 7, 'val'].sum() - \
                        new_df.loc[new_df['m'] == 1, 'new_value'].sum() - \
                        new_df.loc[new_df['m'] == 2, 'new_value'].sum() - \
                        new_df.loc[new_df['m'] == 7, 'new_value'].sum()
            veh_trips = new_df.loc[(new_df['m'] > 2) & (new_df['m'] < 7), 'val'].sum()
            # new_df['final_val'] = new_df['new_value']
            new_df['final_val'] = np.where((new_df['m'] > 2) & (new_df['m'] < 7),
                                           ((new_df['val'] / veh_trips) * sum_trips), new_df['new_value'])

            notem_new_df = notem_new_df.append(new_df)
        print(notem_new_df)
        notem_new_df.to_csv(r"I:\Data\Light Rail\notem_new_msoa_df.csv", index=False)
        return notem_new_df, north_wo_infill

    def _non_tram_infill(self, tram_zone_wo_infill, north_wo_infill, notem_df, notem_new_df, tram_data, notem_tram_seg):
        tram=tram_data.copy()
        tram_data.rename(columns={'val': 'final_val'}, inplace=True)
        df_index_cols = ['p', 'm', 'ca', 'final_val']
        df_group_cols = df_index_cols.copy()
        df_group_cols.remove('final_val')
        notem_df = notem_df.reindex(df_index_cols, axis=1).groupby(df_group_cols).sum().reset_index()
        tram_data = tram_data.reindex(df_index_cols, axis=1).groupby(df_group_cols).sum().reset_index()
        tram_data['final_val'] = 0
        notem_df = notem_df.sort_values(by=['p', 'ca', 'm']).reset_index()
        print(notem_df)
        notem_df[['North', 'Non-Tram', 'Non-Tram adjusted']] = 0.0
        # notem_df['Non-Tram'] = notem_df['North']
        print('raw')
        print(tram_zone_wo_infill)
        print(north_wo_infill)
        index_cols = ['p', 'm', 'ca', 'val']
        group_cols = index_cols.copy()
        group_cols.remove('val')
        tram_zone_wo_infill = tram_zone_wo_infill.reindex(index_cols, axis=1).groupby(
            group_cols).sum().reset_index()
        tram_zone_wo_infill = tram_zone_wo_infill.append(tram_data)
        tram_zone_wo_infill = tram_zone_wo_infill.sort_values(by=['p', 'ca', 'm']).reset_index()
        # notem_df['c_uniq_id'] = notem_df['p'].astype(str) + '_' + notem_df['m'].astype(str) + '_' + notem_df['ca'].astype(str)
        # uniq_id = notem_df['c_uniq_id'].drop_duplicates()
        north_wo_infill = north_wo_infill.append(tram_data)
        north_wo_infill = north_wo_infill.sort_values(by=['p', 'ca', 'm']).reset_index()
        print(tram_zone_wo_infill)
        # uniq_id = [ids for ids in uniq_id if "_7_" not in ids]
        # print(uniq_id)
        # for id in uniq_id:
        #     notem_df.loc[notem_df['c_uniq_id'] == id, 'Non-Tram'] = north_wo_infill['val']-tram_zone_wo_infill['val']
        #     #notem_df['Non-Tram'] = np.where(notem_df[notem_df['c_uniq_id'] == id, (tram_zone_wo_infill['val']-north_wo_infill['val']),0.0)
        notem_df['Non-Tram'] = north_wo_infill['val'] - tram_zone_wo_infill['val']
        notem_df['North'] = notem_new_df['final_val'].to_numpy()
        notem_df['Non-Tram adjusted'] = notem_df['North'] - notem_df['final_val']
        notem_df.rename(columns={'final_val': 'Tram_zones'}, inplace=True)
        notem_df.drop(['index'], axis=1)
        print(notem_df)
        notem_df.to_csv(r"I:\Data\Light Rail\test.csv", index=False)
        internal = pd.read_csv(r"C:\Users\Godzilla\Documents\Internal.csv", )
        total_notem = notem_tram_seg.to_df()
        for col, dt in self._target_col_dtypes['notem'].items():
            total_notem[col] = total_notem[col].astype(dt)
        internal_notem = total_notem.loc[total_notem['msoa_zone_id'].isin(internal['msoa_zone_id'])]
        ntram_notem = internal_notem.loc[~internal_notem['msoa_zone_id'].isin(tram['msoa_zone_id'])]
        ntram_notem['c_uniq_id'] = ntram_notem['msoa_zone_id'].astype(str) + '_' + ntram_notem['p'].astype(str) + '_' + \
                                   ntram_notem['ca'].astype(str)
        uni_id = ntram_notem['c_uniq_id'].drop_duplicates()

        notem_df['c_uniq_id'] = notem_df['p'].astype(str) + '_' + notem_df['ca'].astype(str)
        uniq_id = notem_df['c_uniq_id'].drop_duplicates()
        notem_df_new = pd.DataFrame()
        for id in uniq_id:
            new_df = notem_df.loc[notem_df['c_uniq_id'] == id]

            ntram_sum = new_df.loc[new_df['m'] > 2, 'Non-Tram'].sum()
            ntram_adj_sum = new_df.loc[new_df['m'] > 2, 'Non-Tram adjusted'].sum()
            new_df['ntram_percent'] = np.where((new_df['m'] > 2) & (new_df['m'] < 7),
                                               (new_df['Non-Tram'] / ntram_sum), 0)
            new_df['ntram_adj_percent'] = np.where((new_df['m'] > 2) & (new_df['m'] < 7),
                                                   (new_df['Non-Tram adjusted'] / ntram_adj_sum), 0)
            for ids in uni_id:
                if id in ids:
                    n_df = ntram_notem.loc[ntram_notem['c_uniq_id'] == ids]

                    last_row=n_df.tail(n=1).reset_index()

                    last_row['m']=7
                    last_row['val'] = 0.0

                    n_df=n_df.append(last_row)

                    n_df[['ntram_percent', 'ntram_adj_percent','ntram_new_percent', 'First_adj', 'Final_adj']] = 0.0
                    n_df['ntram_percent'] = new_df['ntram_percent'].to_numpy()
                    n_df['ntram_adj_percent'] = new_df['ntram_adj_percent'].to_numpy()
                    #n_df['First_adj'] = n_df['val'].to_numpy()
                    n_df['First_adj'] = np.where((n_df['m'] > 2) & (n_df['m'] < 7),(n_df['val']*(n_df['ntram_adj_percent'] / n_df['ntram_percent'])), n_df['val'])
                    ntram_new_sum = n_df.loc[n_df['m'] > 2, 'First_adj'].sum()
                    val_sum = n_df.loc[n_df['m'] > 2, 'val'].sum()

                    n_df['ntram_new_percent'] = np.where((n_df['m'] > 2) & (n_df['m'] < 7),
                                                   (n_df['First_adj'] / ntram_new_sum), 0)
                    n_df['Final_adj'] = np.where((n_df['m'] > 2) & (n_df['m'] < 7),(val_sum*n_df['ntram_new_percent']), n_df['First_adj'])

                    notem_df_new = notem_df_new.append(n_df)
                    print(notem_df_new)

                    notem_df_new.to_csv(r"I:\Data\Light Rail\test1.csv", index=False)
