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

from typing import Dict, Tuple

# Third party imports
import pandas as pd
import numpy as np

# local imports
import normits_demand as nd

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
    _sort_msoa = ['msoa_zone_id', 'p', 'ca', 'm']
    _sort_north = ['p', 'ca', 'm']
    _join_cols = ['msoa_zone_id', 'p', 'ca']
    _trip_ends = {'hb_p': "Home Based Productions",
                  'hb_a': "Home Based Attractions",
                  'nhb_p': "Non Home Based Productions",
                  'nhb_a': "Non Home Based Attractions",
                  }

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
                 tram_data_paths: Dict[str, nd.PathLike],
                 notem_outputs: Dict[str, nd.PathLike],
                 export_home: nd.PathLike,
                 ) -> None:
        """

        Parameters
        ----------
        tram_data_paths:
            The path to the tram data.


        """
        # Check that the paths we need exist!
        [file_ops.check_file_exists(x) for x in tram_data_paths.values()]
        [file_ops.check_file_exists(x) for x in notem_outputs.values()]
        file_ops.check_path_exists(export_home)

        # Assign
        self.tram_data_paths = tram_data_paths
        self.notem_outputs = notem_outputs
        self.export_home = export_home

    def run(self,
            verbose: bool = True) -> None:
        """
        Runs the tram inclusion.
        """
        # Initialise timing
        # TODO(BT): Properly integrate logging
        start_time = timing.current_milli_time()
        du.print_w_toggle(
            "Starting Tram inclusion at: %s" % timing.get_datetime(),
            verbose=verbose
        )

        # Run tram inclusion for each trip end model
        for key in self.tram_data_paths.keys():
            trip_end_start = timing.current_milli_time()
            tram_path = self.tram_data_paths[key]
            notem_data = self.notem_outputs[key]
            out_path = os.path.join(self.export_home, self._trip_ends[key])
            file_ops.create_folder(out_path)
            # Read tram and notem output data
            tram_data, notem_tram_seg = self._read_tram_and_notem_data(tram_path, notem_data, verbose)
            tram_msoa, notem_msoa_wo_infill = self._tram_infill_msoa(notem_tram_seg, tram_data, out_path, verbose)
            north_msoa, north_wo_infill = self._tram_infill_north(notem_tram_seg, tram_msoa, out_path, verbose)
            abc = self._non_tram_infill(notem_msoa_wo_infill, north_wo_infill, tram_msoa, north_msoa, tram_data,
                                        notem_tram_seg, out_path, verbose)
            # Print timing for each trip end model
            trip_end_end = timing.current_milli_time()
            time_taken = timing.time_taken(trip_end_start, trip_end_end)
            du.print_w_toggle(
                "%s tram inclusion took: %s\n"
                % (self._trip_ends[key], time_taken),
                verbose=verbose
            )
        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        du.print_w_toggle(
            "Tram inclusion took: %s\n"
            "Finished at: %s" % (time_taken, end_time),
            verbose=verbose
        )

    def _read_tram_and_notem_data(self,
                                  tram_path: nd.PathLike,
                                  notem_data: nd.PathLike,
                                  verbose: bool,
                                  ) -> Tuple[pd.DataFrame, nd.DVector]:
        """
        Reads in the tram and notem data.

        Parameters
        ----------
        verbose:
            If set to True, it will print out progress updates while
        running.

        Returns
        -------
        tram_data:
            Returns the tram data as dataframe.

        notem_tram_seg:
            Returns the notem output dvector in tram segmentation.
        """
        # Define the zoning and segmentations we want to use
        tram_zoning = nd.get_zoning_system('tram')
        tram_seg = nd.get_segmentation_level('hb_p_m7_ca')

        # Reads the tram data
        du.print_w_toggle("Loading the tram data...", verbose=verbose)
        tram_data = file_ops.read_df(
            path=tram_path,
            find_similar=True,
        )
        # Adds tram as mode 7
        cols = list(tram_data)
        cols.insert(2, 'm')
        tram_data['m'] = 7

        for col, dt in self._target_col_dtypes['tram'].items():
            tram_data[col] = tram_data[col].astype(dt)

        tram_data = tram_data.reindex(cols, axis=1)
        tram_data.rename(columns={'trips': 'val'}, inplace=True)
        # Reads the corresponding notem output
        du.print_w_toggle("Loading the notem output data...", verbose=verbose)
        notem_output_dvec = nd.from_pickle(notem_data)
        # Aggregates the dvector to the required segmentation
        notem_tram_seg = notem_output_dvec.aggregate(out_segmentation=nd.get_segmentation_level('hb_p_m_ca'))

        return tram_data, notem_tram_seg

    def _tram_infill_msoa(self,
                          notem_tram_seg: nd.DVector,
                          tram_data: pd.DataFrame,
                          out_path: nd.PathLike,
                          verbose: bool
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tram infill for the msoa's having tram data in the north.

        Parameters
        ----------
        notem_tram_seg:
            DVector containing notem trip end output in revised segmentation(p,m,ca).
            
        tram_data:
            Dataframe containing tram data at msoa level for the north.

        out_path:
            Path where output files need to be stored.

        verbose:
            If set to True, it will print out progress updates while running.

        Returns
        -------
        notem_new_df:
            Returns the dataframe after tram infill.

        notem_msoa_wo_infill:
            Returns the dataframe before tram infill at msoa level.
        """

        # Converts DVector to dataframe
        notem_df = notem_tram_seg.to_df()

        for col, dt in self._target_col_dtypes['notem'].items():
            notem_df[col] = notem_df[col].astype(dt)

        # Retains only msoa zones that have tram data
        notem_df = notem_df.loc[notem_df['msoa_zone_id'].isin(tram_data['msoa_zone_id'])]
        notem_msoa_wo_infill = notem_df.copy()
        # Adds tram data to the notem dataframe
        notem_df = notem_df.append(tram_data)
        # Creates unique id
        notem_df['c_uniq_id'] = pd_utils.str_join_cols(notem_df, self._join_cols)
        notem_df.sort_values(by='msoa_zone_id')
        du.print_w_toggle("Starting tram infill for msoa zones with tram data...", verbose=verbose)
        # Infills tram data
        notem_new_df = self._infill_internal(notem_df=notem_df, sort_order=self._sort_msoa)
        path = os.path.join(out_path, "tram_msoa.csv")
        notem_new_df.to_csv(path, index=False)
        return notem_new_df, notem_msoa_wo_infill

    def _tram_infill_north(self,
                           notem_tram_seg: nd.DVector,
                           tram_msoa: pd.DataFrame,
                           out_path: nd.PathLike,
                           verbose: bool = True
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """
        Tram infill at the internal area (north area) level.

        Parameters
        ----------
        notem_tram_seg:
            DVector containing notem trip end output in revised segmentation(p,m,ca).

        tram_msoa:
            Dataframe containing tram data at msoa level for the north.

        out_path:
            Path where output files need to be stored.

        verbose:
            If set to True, it will print out progress updates while
        running.

        Returns
        -------
        notem_new_df:
            Returns the dataframe after tram infill at north level.

        north_wo_infill:
            Returns the dataframe before tram infill at north level.
        """
        # Converts Dvector to ie sector level
        ie_sectors = nd.get_zoning_system('ie_sector')
        notem_df = notem_tram_seg.translate_zoning(ie_sectors).to_df()

        for col, dt in self._target_col_dtypes['notem_north'].items():
            notem_df[col] = notem_df[col].astype(dt)
        # Retains only internal zones
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
        notem_df['c_uniq_id'] = pd_utils.str_join_cols(notem_df, ['p', 'ca'])
        du.print_w_toggle("Starting tram infill at north level...", verbose=verbose)
        # Infills tram data
        notem_new_df = self._infill_internal(notem_df=notem_df, sort_order=self._sort_north)
        path = os.path.join(out_path, "tram_north.csv")
        notem_new_df.to_csv(path, index=False)
        return notem_new_df, north_wo_infill

    def _non_tram_infill(self,
                         notem_msoa_wo_infill: pd.DataFrame,
                         north_wo_infill: pd.DataFrame,
                         notem_df: pd.DataFrame,
                         notem_new_df: pd.DataFrame,
                         tram_data: pd.DataFrame,
                         notem_tram_seg: nd.DVector,
                         out_path: nd.PathLike,
                         verbose: bool = True,
                         ):
        """
        Tram infill for msoa zones without tram data in the the internal area (north area).

        Parameters
        ----------
        notem_msoa_wo_infill:
            Dataframe containing notem output for msoa zones with tram data before tram infill.

        north_wo_infill:
            Dataframe containing notem output at internal(north) level before tram infill.

        notem_df:
            Dataframe containing notem output for msoa zones with tram data after tram infill.

        notem_new_df:
            Dataframe containing notem output at internal(north) level after tram infill.

        tram_data:
            Dataframe containing tram data at msoa level for the north.

        notem_tram_seg:
            DVector containing notem trip end output in revised segmentation(p,m,ca).

        out_path:
            Path where output files need to be stored.

        verbose:
            If set to True, it will print out progress updates while running.

        Returns
        -------
        notem_df_new:
            Returns the dataframe after tram infill for msoa zones within internal
            area but without tram data.
        """
        du.print_w_toggle("Starting tram adjustment for non tram areas...", verbose=verbose)
        tram = tram_data.copy()
        tram_data.rename(columns={'val': 'final_val'}, inplace=True)
        # Removes msoa level disaggregation
        df_index_cols = ['p', 'm', 'ca', 'final_val']
        df_group_cols = df_index_cols.copy()
        df_group_cols.remove('final_val')
        print('notem_df before grouping',notem_df)
        notem_df = pd_utils.reindex_and_groupby(notem_df, df_index_cols, ['final_val'])
        print('notem_df after grouping', notem_df)
        tram_data = pd_utils.reindex_and_groupby(tram_data, df_index_cols, ['final_val'])
        tram_data['final_val'] = 0
        notem_df = notem_df.sort_values(by=self._sort_north).reset_index()
        print('notem_df',notem_df)
        # Creates new columns to store results
        notem_df[['North', 'Non-Tram', 'Non-Tram adjusted']] = 0.0

        # Removes msoa level disaggregation
        index_cols = self._sort_north + ['val']
        group_cols = index_cols.copy()
        group_cols.remove('val')
        notem_msoa_wo_infill = pd_utils.reindex_and_groupby(notem_msoa_wo_infill, index_cols, ['val'])
        # Adds tram data
        notem_msoa_wo_infill = notem_msoa_wo_infill.append(tram_data)
        notem_msoa_wo_infill = notem_msoa_wo_infill.sort_values(by=self._sort_north).reset_index()
        north_wo_infill = north_wo_infill.append(tram_data)
        north_wo_infill = north_wo_infill.sort_values(by=self._sort_north).reset_index()
        print('notem_msoa_wo_infill',notem_msoa_wo_infill)
        print('notem_new_df',notem_new_df)
        # Calculations for North, Non-tram and non-tram adjusted
        notem_df['Non-Tram'] = north_wo_infill['val'] - notem_msoa_wo_infill['val']
        notem_df['North'] = notem_new_df['final_val'].to_numpy()
        notem_df['Non-Tram adjusted'] = notem_df['North'] - notem_df['final_val']
        notem_df.rename(columns={'final_val': 'Tram_zones'}, inplace=True)
        notem_df.drop(['index'], axis=1)
        print('notem_df',notem_df)
        path = os.path.join(out_path, "non_tram_north.csv")
        notem_df.to_csv(path, index=False)

        # TODO: find a way to bring in internal zone data
        du.print_w_toggle("Starting tram adjustment for non tram zones...", verbose=verbose)
        internal = pd.read_csv(r"C:\Users\Godzilla\Documents\Internal.csv", )
        total_notem = notem_tram_seg.to_df()
        for col, dt in self._target_col_dtypes['notem'].items():
            total_notem[col] = total_notem[col].astype(dt)
        internal_notem = total_notem.loc[total_notem['msoa_zone_id'].isin(internal['msoa_zone_id'])]
        ntram_notem = internal_notem.loc[~internal_notem['msoa_zone_id'].isin(tram['msoa_zone_id'])]
        ntram_notem['c_uniq_id'] = pd_utils.str_join_cols(ntram_notem, self._join_cols)
        uni_id = ntram_notem['c_uniq_id'].drop_duplicates()

        notem_df['c_uniq_id'] = pd_utils.str_join_cols(notem_df, ['p', 'ca'])
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
                    print(id)
                    n_df = ntram_notem.loc[ntram_notem['c_uniq_id'] == ids]

                    last_row = n_df.tail(n=1).reset_index()

                    last_row['m'] = 7
                    last_row['val'] = 0.0

                    n_df = n_df.append(last_row)

                    n_df[['ntram_percent', 'ntram_adj_percent', 'ntram_new_percent', 'First_adj', 'Final_adj']] = 0.0
                    n_df['ntram_percent'] = new_df['ntram_percent'].to_numpy()
                    n_df['ntram_adj_percent'] = new_df['ntram_adj_percent'].to_numpy()

                    n_df['First_adj'] = np.where((n_df['m'] > 2) & (n_df['m'] < 7),
                                                 (n_df['val'] * (n_df['ntram_adj_percent'] / n_df['ntram_percent'])),
                                                 n_df['val'])
                    ntram_new_sum = n_df.loc[n_df['m'] > 2, 'First_adj'].sum()
                    val_sum = n_df.loc[n_df['m'] > 2, 'val'].sum()

                    n_df['ntram_new_percent'] = np.where((n_df['m'] > 2) & (n_df['m'] < 7),
                                                         (n_df['First_adj'] / ntram_new_sum), 0)
                    n_df['Final_adj'] = np.where((n_df['m'] > 2) & (n_df['m'] < 7),
                                                 (val_sum * n_df['ntram_new_percent']), n_df['First_adj'])

                    notem_df_new = notem_df_new.append(n_df)
                    # print(notem_df_new)

        path_new = os.path.join(out_path, "non_tram_msoa.csv")
        notem_new_df.to_csv(path_new, index=False)

    def _infill_internal(self,
                         notem_df: pd.DataFrame,
                         sort_order: list,
                         ) -> pd.DataFrame:
        """
        Creates a subset of the given dataframe using a condition.

        Parameters
        ----------
        notem_df:
            The original dataframe that needs to be subset.

        sort_order:
            Condition for slicing

        Returns
        -------
        notem_sub_df:
            Returns the subset of the original dataframe.

        """
        # create subset of the notem dataframe for train and tram modes
        notem_train = self._create_df_subset(notem_df=notem_df, mode=6)
        notem_tram = self._create_df_subset(notem_df=notem_df, mode=7)
        # Checks whether train trips are higher and if so subtracts tram trips from train trips
        notem_train['new_value'] = np.where(
            (notem_train['c_uniq_id'].isin(notem_tram['c_uniq_id'])) & (notem_train['val'] > notem_tram['val']),
            (notem_train['val'] - notem_tram['val']), 0)
        # Checks whether tram trips are higher and if so curtails them to train trips
        notem_tram['new_value'] = np.where(
            (notem_train['c_uniq_id'].isin(notem_tram['c_uniq_id'])) & (notem_train['val'] > notem_tram['val']),
            notem_tram['val'], notem_train['val'])

        notem_df = notem_df[notem_df.m < 6]
        notem_df['new_value'] = notem_df['val']

        notem_df = notem_df.append([notem_train, notem_tram])

        notem_df.drop(columns='index', inplace=True)

        notem_df = notem_df.sort_values(by=sort_order)
        notem_df.to_csv(r"I:\Data\Light Rail\notem_df.csv", index=False)

        uniq_id = notem_df['c_uniq_id'].drop_duplicates()
        notem_new_df = pd.DataFrame()

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
        return notem_new_df

    def _create_df_subset(self,
                          notem_df: pd.DataFrame,
                          mode: int,
                          ) -> pd.DataFrame:
        """
        Creates a subset of the given dataframe using a condition.

        Parameters
        ----------
        notem_df:
            The original dataframe that needs to be subset.

        mode:
            Condition for slicing

        Returns
        -------
        notem_sub_df:
            Returns the subset of the original dataframe.

        """
        notem_sub_df = notem_df.loc[notem_df['m'] == mode]
        notem_sub_df = notem_sub_df.sort_values(by='c_uniq_id')
        notem_sub_df.reset_index(inplace=True)
        return notem_sub_df
