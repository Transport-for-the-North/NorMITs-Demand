# -*- coding: utf-8 -*-
"""
Created on: 09/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Classes which build all the paths for TMS model inputs outputs
"""
# Built-Ins
import os
import abc
import collections

from typing import Dict

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand.utils import file_ops
from normits_demand.utils import pandas_utils as pd_utils


class ExternalModelArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the argument builder for the external model should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions to pick up new import files.
    """

    @abc.abstractmethod
    def build_hb_external_model_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass

    @abc.abstractmethod
    def build_nhb_external_model_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass


class GravityModelArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the argument builder for the gravity model should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions to pick up new import files.
    """

    @abc.abstractmethod
    def build_hb_gravity_model_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass

    @abc.abstractmethod
    def build_nhb_gravity_model_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass


class ExternalModelArgumentBuilder(ExternalModelArgumentBuilderBase):
    # Costs constants
    _cost_type = '24hr'
    _cost_dir_name = 'costs'
    _cost_base_fname = "{zoning_name}_{cost_type}_costs.csv"

    # CJTW constants
    _cjtw_infill = 0.1
    _cjtw_dir_name = 'cjtw'
    _cjtw_base_fname = 'cjtw_{zoning_name}.csv'

    # Trip Length Distribution constants
    _tld_dir_name = 'trip_length_distributions'
    _tld_area_dir_name = 'gb'

    def __init__(self,
                 import_home: nd.PathLike,
                 base_year: int,
                 scenario: str,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 internal_tld_name: str,
                 external_tld_name: str,
                 hb_running_segmentation: nd.core.segments.SegmentationLevel,
                 nhb_running_segmentation: nd.core.segments.SegmentationLevel,
                 notem_iteration_name: str,
                 notem_export_home: str,
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # TODO(BT): Validate segments and zones are the correct types

        # Assign attributes
        self.import_home = import_home
        self.zoning_system = zoning_system
        self.internal_tld_name = internal_tld_name
        self.external_tld_name = external_tld_name
        self.hb_running_segmentation = hb_running_segmentation
        self.nhb_running_segmentation = nhb_running_segmentation

        # Generate the NoTEM export paths
        self.notem_exports = nd.pathing.NoTEMExportPaths(
            path_years=[base_year],
            scenario=scenario,
            iteration_name=notem_iteration_name,
            export_home=notem_export_home,
        )

    def _read_and_convert_cjtw(self):
        # Build a path to the input
        fname = self._cjtw_base_fname.format(zoning_name=self.zoning_system.name)
        path = os.path.join(self.import_home, self._cjtw_dir_name, fname)

        # Read and convert to zoning system format
        cjtw = read_cjtw(
            file_path=path,
            zoning_name=self.zoning_system.name,
            subset=None,
            reduce_to_pa_factors=False,
        )

        # Aggregate mode
        p_col = list(cjtw)[0]
        a_col = list(cjtw)[1]
        cjtw = cjtw.reindex([p_col, a_col, 'trips'], axis=1)
        cjtw = cjtw.groupby([p_col, a_col]).sum().reset_index()

        # Convert to a wide matrix
        return pd_utils.long_to_wide_infill(
            df=cjtw,
            index_col=p_col,
            columns_col=a_col,
            values_col='trips',
            index_vals=self.zoning_system.unique_zones,
            column_vals=self.zoning_system.unique_zones,
            infill=self._cjtw_infill,
        )

    def build_external_model_arguments(self,
                                       trip_origin: str,
                                       ) -> Dict[str, nd.PathLike]:
        # Init
        trip_origin = trip_origin.lower()

        # ## READ IN PRODUCTIONS AND ATTRACTIONS ## #
        if trip_origin == 'hb':
            productions_path = self.notem_exports.hb_production.export_paths.notem_segmented
            attractions_path = self.notem_exports.hb_attraction.export_paths.notem_segmented
            running_segmentation = self.hb_running_segmentation
        elif trip_origin == 'nhb':
            productions_path = self.notem_exports.nhb_production.export_paths.notem_segmented
            attractions_path = self.notem_exports.nhb_attraction.export_paths.notem_segmented
            running_segmentation = self.nhb_running_segmentation
        else:
            raise ValueError(
                "Received an unexpected value for trip origin. Expected one of "
                "'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        # TODO(BT): Tidy up this function - no cache!
        productions, attractions = import_pa(
            production_import_path=productions_path,
            attraction_import_path=attractions_path,
            model_zone=self.zoning_system.name,
            trip_origin=trip_origin,
        )

        # Build TLD directory paths
        base_tld_path = os.path.join(
            self.import_home,
            self._tld_dir_name,
            self._tld_area_dir_name,
        )
        internal_tld_path = os.path.join(base_tld_path, self.internal_tld_name)
        external_tld_path = os.path.join(base_tld_path, self.external_tld_name)

        # Build costs path
        fname = self._cost_base_fname.format(
            zoning_name=self.zoning_system.name,
            cost_type=self._cost_type,
        )
        costs_path = os.path.join(
            self.import_home,
            self.zoning_system.name,
            self._cost_dir_name,
            fname,
        )

        # Check paths exist
        paths = [
            internal_tld_path,
            external_tld_path,
            costs_path,
        ]

        for path in paths:
            if not os.path.exists(path):
                raise IOError(
                    "Generated path doesn't exist!\nGenerated the following "
                    "path that does not exist while building External "
                    "Model arguments.\n %s"
                    % path
                )

        # Return the generated arguments
        return {
            'productions': productions,
            'attractions': attractions,
            'seed_matrix': self._read_and_convert_cjtw(),
            'costs_path': costs_path,
            'internal_tld_dir': internal_tld_path,
            'external_tld_dir': external_tld_path,
            'running_segmentation': running_segmentation,
        }

    def build_hb_external_model_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_external_model_arguments(trip_origin='hb')

    def build_nhb_external_model_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_external_model_arguments(trip_origin='nhb')


class GravityModelArgumentBuilder(GravityModelArgumentBuilderBase):
    def __init__(self):
        pass

    def build_hb_gravity_model_arguments(self) -> Dict[str, nd.PathLike]:
        raise NotImplementedError()

    def build_nhb_gravity_model_arguments(self) -> Dict[str, nd.PathLike]:
        raise NotImplementedError()


class TMSExportPaths:
    # TODO(BT): Finalise TMS exports structure

    def __init__(self):
        # We would assign export paths here
        pass


class ExternalModelExportPaths:
    # Constants - some for TMS parent?
    _productions_dir_name = 'Productions'
    _attractions_dir_name = 'Attractions'
    _distributions_dir_name = 'Attractions'

    _int_productions_base_name = '{trip_origin}_internal_productions.csv'
    _int_attractions_base_name = '{trip_origin}_internal_attractions.csv'

    # Report dir names
    _log_dir_name = 'Logs'
    _tld_report_dir = 'TLD Reports'
    _ie_report_dir = 'IE Reports'

    # Output path classes
    ExportPaths = collections.namedtuple(
        typename='ExportPaths',
        field_names=[
            'home',
            'hb_internal_productions',
            'nhb_internal_productions',
            'hb_internal_attractions',
            'nhb_internal_attractions',
            'external_distribution_dir',
        ]
    )

    ReportPaths = collections.namedtuple(
        typename='ReportPaths',
        field_names=[
            'home',
            'model_log_dir',
            'tld_report_dir',
            'ie_report_dir',
        ]
    )

    def __init__(self,
                 export_home: nd.PathLike,
                 report_home: nd.PathLike,
                 ):
        # Assign attributes
        self.export_home = export_home  # Something like I:\NorMITs Demand\noham\TMS\iter8\External Model
        self.report_home = report_home

        # Make sure paths exist
        try:
            file_ops.check_path_exists(export_home)
            file_ops.check_path_exists(report_home)
        except IOError as e:
            raise type(e)(
                "Got the following error while checking if the export_home and "
                "report_home paths exist:\n%s"
                % str(e)
            )

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """Creates self.export_paths"""

        # Generate production and paths
        production_out = os.path.join(self.export_home, self._productions_dir_name)

        fname = self._int_productions_base_name.format(trip_origin='hb')
        hb_internal_productions = os.path.join(production_out, fname)

        fname = self._int_productions_base_name.format(trip_origin='nhb')
        nhb_internal_productions = os.path.join(production_out, fname)

        # Generate attraction and paths
        attraction_out = os.path.join(self.export_home, self._attractions_dir_name)

        fname = self._int_attractions_base_name.format(trip_origin='hb')
        hb_internal_attractions = os.path.join(attraction_out, fname)

        fname = self._int_attractions_base_name.format(trip_origin='nhb')
        nhb_internal_attractions = os.path.join(attraction_out, fname)

        # Generate external distribution path
        external_distribution_dir = os.path.join(
            self.export_home,
            self._distributions_dir_name,
        )
        file_ops.create_folder(external_distribution_dir)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            hb_internal_productions=hb_internal_productions,
            nhb_internal_productions=nhb_internal_productions,
            hb_internal_attractions=hb_internal_attractions,
            nhb_internal_attractions=nhb_internal_attractions,
            external_distribution_dir=external_distribution_dir,
        )

    def _create_report_paths(self) -> None:
        """Creates self.report_paths"""

        # Create the export_paths class
        self.report_paths = self.ReportPaths(
            home=self.report_home,
            model_log_dir=os.path.join(self.report_home, self._log_dir_name),
            tld_report_dir=os.path.join(self.report_home, self._tld_report_dir),
            ie_report_dir=os.path.join(self.report_home, self._ie_report_dir),
        )

        # Make paths that don't exist
        for path in self.report_paths:
            file_ops.create_folder(path)


def import_pa(production_import_path,
              attraction_import_path,
              model_zone,
              trip_origin,
              ):
    """
    This function imports productions and attractions from given paths.

    Parameters
    ----------
    production_import_path:
        Path to import productions from.

    attraction_import_path:
        Path to import attractions from.

    model_zone:
        Type of model zoning system. norms or noham

    Returns
    ----------
    [0] productions:
        Mainland GB productions.

    [1] attractions:
        Mainland GB attractions.
    """
    p_cache = "E:/%s_productions.csv" % model_zone
    a_cache = "E:/%s_attractions.csv" % model_zone

    if os.path.exists(p_cache) and os.path.exists(a_cache):
        return pd.read_csv(p_cache, index_col=0), pd.read_csv(a_cache, index_col=0)

    # Reading pickled Dvector
    prod_dvec = nd.from_pickle(production_import_path)

    # Aggregate to the required segmentation
    if trip_origin == 'hb':
        if model_zone == 'noham':
            agg_seg = nd.get_segmentation_level('hb_p_m_6tp')
        elif model_zone == 'norms':
            agg_seg = nd.get_segmentation_level('hb_p_m_ca_6tp')
        else:
            raise ValueError("Invalid model name")
    elif trip_origin == 'nhb':
        if model_zone == 'noham':
            agg_seg = nd.get_segmentation_level('nhb_p_m_6tp')
        elif model_zone == 'norms':
            agg_seg = nd.get_segmentation_level('nhb_p_m_ca_6tp')
        else:
            raise ValueError("Invalid model name")
    else:
        raise ValueError("Invalid trip origin")

    # Aggregate and translate for norms/noham
    prod_dvec_agg = prod_dvec.aggregate(out_segmentation=agg_seg)
    model_zoning = nd.get_zoning_system(model_zone)
    prod_dvec = prod_dvec_agg.translate_zoning(model_zoning, "population")

    # Weekly trips to weekday trips conversion
    prod_df = prod_dvec.to_df()
    prod_wd = weekly_to_weekday(prod_df, trip_origin, model_zone)

    # Reading pickled Dvector
    attr_dvec = nd.from_pickle(attraction_import_path)

    # Aggregate and translate for norms/noham
    attr_dvec_agg = attr_dvec.aggregate(out_segmentation=agg_seg)
    model_zoning = nd.get_zoning_system(model_zone)
    attr_dvec = attr_dvec_agg.translate_zoning(model_zoning, "employment")

    # Weekly trips to weekday trips conversion
    attr_df = attr_dvec.to_df()
    attr_wd = weekly_to_weekday(attr_df, trip_origin, model_zone)

    # TODO(BT): Sort zoning system into order
    prod_wd.to_csv(p_cache)
    attr_wd.to_csv(a_cache)

    return prod_wd, attr_wd


def weekly_to_weekday(df, trip_origin, model_zone) -> pd.DataFrame:
    """
    Convert weekly trips to weekday trips.

    Removes tp5 and tp6 from the time period column and
    divides trips by 5 to convert them from weekly to weekday.

    Parameters
    ----------
    df:
    Dataframe (either productions or attractions) containing notem segmented weekly trips.

    trip_origin:
    Whether the trip origin is hb or nhb.

    Return
    ----------
    df:
    Dataframe (either productions or attractions) containing notem segmented weekday trips.
    """
    if model_zone == 'norms':
        df[["p", "m", "ca", "tp"]] = df[["p", "m", "ca", "tp"]].apply(pd.to_numeric)
    else:
        df[["p", "m", "tp"]] = df[["p", "m", "tp"]].apply(pd.to_numeric)
    df = df.drop(df[df.tp >= 5].index)
    df['val'] = df['val'] / 5
    df_index_cols = list(df)
    df_index_cols.remove('tp')
    df_group_cols = df_index_cols.copy()
    df_group_cols.remove('val')

    # Time period removed for hb based trips
    if trip_origin == 'hb':
        df = df.reindex(df_index_cols, axis=1).groupby(df_group_cols).sum().reset_index()
    return df


def read_cjtw(file_path: nd.PathLike,
              zoning_name: str,
              subset: bool = None,
              reduce_to_pa_factors: bool = True,
              ) -> pd.DataFrame:
    """
    This function imports census journey to work and converts types
    to ntem journey types

    Parameters
    ----------
    file_path:
        Takes a model folder to look for a cjtw zonal conversion

    zoning_name:
        The name of the zoning system the cjtw file is in

    subset:
        Takes a vector of model zones to filter by. Mostly for test model runs.

    reduce_to_pa_factors:
        ???

    Returns
    ----------
    census_journey_to_work:
        A census journey to work distribution in the required zonal format.
    """
    # TODO(BT, CS): Re-write this to be more generic
    # Init
    zoning_name = zoning_name.lower()

    # Read in the file
    if not os.path.isfile(file_path):
        raise ValueError("No file exists at %s" % file_path)
    cjtw = pd.read_csv(file_path)

    # CTrip End Categories
    # 1 Walk
    # 2 Cycle
    # 3 Car driver
    # 4 Car passenger
    # 5 Bus
    # 6 Rail / underground

    if subset is not None:
        sub_col = list(subset)
        sub_zones = subset[sub_col].squeeze()
        cjtw = cjtw[cjtw['1_' + zoning_name + 'Areaofresidence'].isin(sub_zones)]
        cjtw = cjtw[cjtw['2_' + zoning_name + 'Areaofworkplace'].isin(sub_zones)]

    method_to_mode = {'4_Workmainlyatorfromhome': '1_walk',
                      '5_Undergroundmetrolightrailtram': '6_rail_ug',
                      '6_Train': '6_rail_ug',
                      '7_Busminibusorcoach': '5_bus',
                      '8_Taxi': '3_car',
                      '9_Motorcyclescooterormoped': '2_cycle',
                      '10_Drivingacarorvan': '3_car',
                      '11_Passengerinacarorvan': '3_car',
                      '12_Bicycle': '2_cycle',
                      '13_Onfoot': '1_walk',
                      '14_Othermethodoftraveltowork': '1_walk'}
    modeCols = list(method_to_mode.keys())

    for col in modeCols:
        cjtw = cjtw.rename(columns={col: method_to_mode.get(col)})

    cjtw = cjtw.drop('3_Allcategories_Methodoftraveltowork', axis=1)
    cjtw = cjtw.groupby(cjtw.columns, axis=1).sum()
    cjtw = cjtw.reindex(['1_' + zoning_name + 'Areaofresidence',
                         '2_' + zoning_name + 'Areaofworkplace',
                         '1_walk', '2_cycle', '3_car',
                         '5_bus', '6_rail_ug'], axis=1)
    # Redefine mode cols for new aggregated modes
    modeCols = ['1_walk', '2_cycle', '3_car', '5_bus', '6_rail_ug']
    # Pivot
    cjtw = pd.melt(cjtw, id_vars=['1_' + zoning_name + 'Areaofresidence',
                                  '2_' + zoning_name + 'Areaofworkplace'],
                   var_name='mode', value_name='trips')
    cjtw['mode'] = cjtw['mode'].str[0]

    # Build distribution factors
    hb_totals = cjtw.drop(
        '2_' + zoning_name + 'Areaofworkplace',
        axis=1
    ).groupby(
        ['1_' + zoning_name + 'Areaofresidence', 'mode']
    ).sum().reset_index()

    hb_totals = hb_totals.rename(columns={'trips': 'zonal_mode_total_trips'})
    hb_totals = hb_totals.reindex(
        ['1_' + zoning_name + 'Areaofresidence', 'mode', 'zonal_mode_total_trips'],
        axis=1
    )

    cjtw = cjtw.merge(hb_totals,
                      how='left',
                      on=['1_' + zoning_name + 'Areaofresidence', 'mode'])

    # Divide by total trips to get distribution factors

    if reduce_to_pa_factors:
        cjtw['distribution'] = cjtw['trips'] / cjtw['zonal_mode_total_trips']
        cjtw = cjtw.drop(['trips', 'zonal_mode_total_trips'], axis=1)
    else:
        cjtw = cjtw.drop(['zonal_mode_total_trips'], axis=1)

    return cjtw
