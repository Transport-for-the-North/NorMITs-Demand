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

from typing import Dict

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd


class TMSArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the import paths class for TMS should look.

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


class TMSArgumentBuilder(TMSArgumentBuilderBase):
    # Constants
    _cost_type = '24hr'

    def __init__(self,
                 base_year: int,
                 scenario: str,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 notem_iteration_name: str,
                 notem_export_home: str,
                 ):
        # Assign attributes
        self.zoning_system = zoning_system

        # Generate the NoTEM export paths
        self.notem_exports = nd.pathing.NoTEMExportPaths(
            path_years=base_year,
            scenario=scenario,
            iteration_name=notem_iteration_name,
            export_home=notem_export_home,
        )

    def build_external_model_arguments(self,
                                       trip_origin: str,
                                       ) -> Dict[str, nd.PathLike]:
        # Init
        trip_origin = trip_origin.lower()

        # ## READ IN PRODUCTIONS AND ATTRACTIONS ## #
        if trip_origin == 'hb':
            productions_path = self.notem_exports.hb_production.notem_segmented
            attractions_path = self.notem_exports.hb_attraction.notem_segmented
        elif trip_origin == 'nhb':
            productions_path = self.notem_exports.nhb_production.notem_segmented
            attractions_path = self.notem_exports.nhb_attraction.notem_segmented
        else:
            raise ValueError(
                "Received an unexpected value for trip origin. Expected one of "
                "'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        productions, attractions = import_pa(
            production_import_path=productions_path,
            attraction_import_path=attractions_path,
            model_zone=self.zoning_system.name,
            trip_origin=trip_origin,
        )


        # ## TIDY THIS UP - PULLING ALL PATHS OUT TO HERE ## #
        # TLDS
        TLD_HOME = r"I:\NorMITs Synthesiser\import\trip_length_bands"
        tld_dir = os.path.join(TLD_HOME, params['tld_area'])
        internal_tld_dir = os.path.join(tld_dir, params['internal_tld_bands'])
        external_tld_dir = os.path.join(tld_dir, params['external_tld_bands'])

        # ## GET CJTW ## #
        print('Importing cjtw...')
        cjtw = nup.get_cjtw(self.lookup_folder,
                            self.params['model_zoning'].lower(),
                            subset=None,
                            reduce_to_pa_factors=False)
        # Aggregate mode
        p_col = list(cjtw)[0]
        a_col = list(cjtw)[1]
        cjtw = cjtw.reindex([p_col, a_col, 'trips'], axis=1)
        cjtw = cjtw.groupby([p_col, a_col]).sum().reset_index()

        # Handle cols - transpose
        print('cjtw:\n %s\n' % cjtw)
        unq_zones = nup.get_zone_range(productions[self.zone_col])
        cjtw = nup.df_to_np(cjtw,
                            v_heading=p_col,
                            h_heading=a_col,
                            values='trips',
                            unq_internal_zones=unq_zones)

        # Small infill
        seed_infill = 0.1
        cjtw = np.where(cjtw == 0, seed_infill, cjtw)

        # costs
        # path = model_zone_lookup + costs
        reports_dir = self.tms_out['reports']

        ext = em.ExternalModel(
            zoning_system,
        )


        return {
            'cost_type': self._cost_type,
            'productions': productions,
            'attractions': attractions,
            'seed_matrix': cjtw,
            'costs_dir': costs_dir,
            'reports_dir': reports_dir,
            'internal_tld_dir': internal_tld_dir,
            'external_tld_dir': external_tld_dir,
        }

    def build_hb_external_model_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_external_model_arguments(trip_origin='hb')

    def build_nhb_external_model_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_external_model_arguments(trip_origin='nhb')

    def build_hb_gravity_model_arguments(self) -> Dict[str, nd.PathLike]:
        raise NotImplementedError()

    def build_nhb_gravity_model_arguments(self) -> Dict[str, nd.PathLike]:
        raise NotImplementedError()


class TMSExportPaths:
    # TODO(BT): Finalise TMS exports structure

    def __init__(self):
        # We would assign export paths here
        pass


def import_pa(production_import_path,
              attraction_import_path,
              model_zone,
              trip_origin):
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
        return pd.read_csv(p_cache), pd.read_csv(a_cache)

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
