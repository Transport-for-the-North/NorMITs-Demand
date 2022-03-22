# -*- coding: utf-8 -*-
"""
Created on: 07/12/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
from __future__ import annotations

# Built-Ins
import os
import abc
import collections

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

# Third Party
import numpy as np
import pandas as pd
import tqdm

# Local Imports
import normits_demand as nd
from normits_demand import constants
from normits_demand.cost import utils as cost_utils
from normits_demand.utils import file_ops
from normits_demand.utils import translation
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

# ## DEFINE COLLECTIONS OF OUTPUT PATHS ## #
# Exports
_DM_ExportPaths_NT = collections.namedtuple(
    typename='_DM_ExportPaths_NT',
    field_names=[
        'home',
        'upper_external_pa',
        'full_pa_dir',
        'compiled_pa_dir',
        'full_od_dir',
        'compiled_od_dir',
        'compiled_od_dir_pcu',
    ]
)


_DistributorExportPaths_NT = collections.namedtuple(
    typename='_DistributorExportPaths_NT',
    field_names=[
        'home',
        'matrix_dir',
    ]
)

# Reports
_DM_ReportPaths_NT = collections.namedtuple(
    typename='_DM_ReportPaths_NT',
    field_names=[
        'home',
        'pa_reports_dir',
        'od_reports_dir',
        'lower_vector_reports_dir',
    ]
)


_DistributorReportPaths_NT = collections.namedtuple(
    typename='_DistributorReportPaths_NT',
    field_names=[
        'home',
        'overall_log',
        'model_log_dir',
        'tld_report_dir',
    ]
)


# ## DEFINE IMPORT PATHS ## #
class DMArgumentBuilderBase(abc.ABC):
    """
    Abstract Class defining how the argument builder for the
    distribution model should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. DistributionModel can then use the
    defined functions/properties to pick up new import files.
    """

    _translation_weight_col = 'weight'
    _external_suffix = 'ext'

    # Cache
    _production_base_cache = '{trip_origin}p_{zoning}_{mode}_{tier}_cache.pbz2'
    _attraction_base_cache = '{trip_origin}a_{zoning}_{mode}_{tier}_cache.pbz2'

    # Lower vector report filenames
    _segment_totals_bname = '{trip_origin}_{vec_name}_lower_vector_{year}_segment_totals.csv'
    _ca_sector_bname = '{trip_origin}_{vec_name}_lower_vector_{year}_ca_sector_totals.csv'
    _ie_sector_bname = '{trip_origin}_{vec_name}_lower_vector_{year}_ie_sector_totals.csv'

    def __init__(self,
                 year: int,
                 trip_origin: str,
                 running_mode: nd.Mode,
                 running_segmentation: nd.SegmentationLevel,
                 upper_zoning_system: nd.ZoningSystem,
                 upper_running_zones: List[Any],
                 lower_zoning_system: nd.ZoningSystem,
                 lower_running_zones: List[Any],
                 cache_path: nd.PathLike = None,
                 overwrite_cache: nd.PathLike = False,
                 ):
        self.year = year
        self.trip_origin = trip_origin
        self.running_mode = running_mode
        self.running_segmentation = running_segmentation
        self.upper_zoning_system = upper_zoning_system
        self.upper_running_zones = upper_running_zones
        self.lower_zoning_system = lower_zoning_system
        self.lower_running_zones = lower_running_zones

        self.cache_path = cache_path
        self.overwrite_cache = overwrite_cache

    def _get_translations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the translations between upper and lower zoning"""
        return translation.get_long_pop_emp_translations(
            in_zoning_system=self.upper_zoning_system,
            out_zoning_system=self.lower_zoning_system,
            weight_col_name=self._translation_weight_col
        )

    def _save_external_demand(self,
                              df: pd.DataFrame,
                              segment_params: Dict[str, Any],
                              external_matrix_output_dir: nd.PathLike,
                              ) -> None:
        """Write the demand to disk"""
        # Generate the output path
        fname = self.running_segmentation.generate_file_name(
            segment_params=segment_params,
            file_desc='synthetic_pa',
            trip_origin=self.trip_origin,
            year=self.year,
            suffix=self._external_suffix,
            compressed=True
        )
        out_path = os.path.join(external_matrix_output_dir, fname)

        # Write out
        file_ops.write_df(df, out_path)

    def _report_vector(self,
                       df: pd.DataFrame,
                       df_name: str,
                       report_dir: nd.PathLike,
                       ) -> None:

        # Convert to Dvec
        dvec = nd.DVector(
            zoning_system=self.lower_zoning_system,
            segmentation=self.running_segmentation,
            import_data=df,
            zone_col=self.lower_zoning_system.col_name,
            val_col='val',
            time_format='avg_day',
        )

        # Generate filenames
        kwargs = {
            'trip_origin': self.trip_origin,
            'vec_name': df_name,
            'year': self.year,
        }
        segment_totals_fname = self._segment_totals_bname.format(**kwargs)
        ca_sector_fname = self._ca_sector_bname.format(**kwargs)
        ie_sector_fname = self._ie_sector_bname.format(**kwargs)

        # Generate and write reports
        dvec.write_sector_reports(
            segment_totals_path=os.path.join(report_dir, segment_totals_fname),
            ca_sector_path=os.path.join(report_dir, ca_sector_fname),
            ie_sector_path=os.path.join(report_dir, ie_sector_fname),
        )

    def _convert_upper_pa_to_lower(self,
                                   upper_model_matrix_dir: nd.PathLike,
                                   external_matrix_output_dir: nd.PathLike,
                                   lower_model_vector_report_dir: nd.PathLike,
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Converts Upper matrices into vectors for lower model

        Parameters
        ----------
        upper_model_matrix_dir:
            The directory containing the upper model's output matrices

        external_matrix_output_dir:
            The directory to output all of the external demand, in the
            lower zoning system. I.E. All demand in lower_zoning_system,
            that is not also in the lower_running_zones.

        lower_model_vector_report_dir:
            The directory to output standard reports of the vectors generated
            for the lower model. This is all demand in the lower_running_zones,
            and all data that is not in the external_matrix_output_dir.

        Returns
        -------
        productions:
            pandas DataFrame of productions. Will have columns named after:
            self.lower_zoning_system.name, segment_names, and 'productions'

        attractions:
            pandas DataFrame of attractions. Will have columns named after:
            self.lower_zoning_system.name, segment_names, and 'attractions'
        """
        # Init
        in_zone_col = self.upper_zoning_system.col_name
        out_zone_col = self.lower_zoning_system.col_name

        # Don't need any translations if same
        if self.upper_zoning_system != self.lower_zoning_system:
            pop_trans, emp_trans = self._get_translations()
        else:
            pop_trans = None
            emp_trans = None

        # Convert upper matrices into a efficient dataframes
        eff_df_list = list()
        segment_col_names = set()
        desc = "Converting Upper Demand for Lower Model"
        total = len(self.running_segmentation)
        for segment_params in tqdm.tqdm(self.running_segmentation, desc=desc, total=total):
            # Read in DF
            fname = self.running_segmentation.generate_file_name(
                trip_origin=self.trip_origin,
                year=str(self.year),
                file_desc='synthetic_pa',
                segment_params=segment_params,
                compressed=True,
            )
            path = os.path.join(upper_model_matrix_dir, fname)
            df = file_ops.read_df(path, index_col=0)

            # Make sure index and columns are the same type
            df.columns = df.columns.astype(df.index.dtype)

            # Translate to lower zoning
            if not(pop_trans is None or emp_trans is None):
                df = translation.pandas_matrix_zone_translation(
                    matrix=df,
                    row_translation=pop_trans,
                    col_translation=emp_trans,
                    from_zone_col=self.upper_zoning_system.col_name,
                    to_zone_col=self.lower_zoning_system.col_name,
                    factors_col=self._translation_weight_col,
                    from_unique_zones=self.upper_zoning_system.unique_zones,
                    to_unique_zones=self.lower_zoning_system.unique_zones,
                )

            # Split into the internal and external demand for lower model
            internal_mask = pd_utils.get_wide_mask(df=df, zones=self.lower_running_zones)
            lower_model_demand = df * internal_mask
            external_demand = df * ~internal_mask

            # Save the demand that isn't going into the lower model
            self._save_external_demand(
                df=external_demand,
                segment_params=segment_params,
                external_matrix_output_dir=external_matrix_output_dir,
            )

            # Keep track of the column names we're keeping
            seg_cols = list(segment_params.keys())
            segment_col_names = set(list(segment_col_names) + seg_cols)

            # Convert to production and attraction vectors
            index_col = lower_model_demand.index
            index_col.name = out_zone_col

            productions = pd.DataFrame(
                data=lower_model_demand.values.sum(axis=1),
                index=index_col,
                columns=['productions'],
            )
            attractions = pd.DataFrame(
                data=lower_model_demand.values.sum(axis=0),
                index=index_col,
                columns=['attractions'],
            )

            # Stick into an efficient DF
            eff_df = segment_params.copy()
            eff_df['df'] = productions.join(attractions).reset_index()
            eff_df_list.append(eff_df)

        # Compile the efficient DFs
        segment_col_names = [out_zone_col] + list(segment_col_names)
        final_cols = segment_col_names + ['productions', 'attractions']
        vector = du.compile_efficient_df(eff_df_list, col_names=final_cols)
        vector = vector.sort_values(by=segment_col_names).reset_index(drop=True)

        productions = vector.drop(columns=['attractions'])
        productions = productions.rename(columns={'productions': 'val'})

        attractions = vector.drop(columns=['productions'])
        attractions = attractions.rename(columns={'attractions': 'val'})

        # Generate standard vector reports
        self._report_vector(productions, 'productions', lower_model_vector_report_dir)
        self._report_vector(attractions, 'attractions', lower_model_vector_report_dir)

        return productions, attractions

    def _get_latest_matrix_time(self, matrix_dir: nd.PathLike) -> float:
        """Gets the latest modified time from all the relevant matrices

        Parameters
        ----------
        matrix_dir:
            The directory to check the matrices in

        Returns
        -------
        timestamp:
            A floating point number giving the number of seconds since the
            epoch (see the time module) for the most recent matrix file in
            matrix_dir.

        Raises
        ------
        OSError:
            if the file does not exist or is inaccessible.
        """
        latest_time = 0
        for segment_params in self.running_segmentation:
            # Read in DF
            fname = self.running_segmentation.generate_file_name(
                trip_origin=self.trip_origin,
                year=str(self.year),
                file_desc='synthetic_pa',
                segment_params=segment_params,
                compressed=True,
            )
            path = os.path.join(matrix_dir, fname)

            # Keep the latest time
            mat_modified_time = os.path.getmtime(path)
            if mat_modified_time > latest_time:
                latest_time = mat_modified_time

        return latest_time

    @staticmethod
    def _get_latest_cache_time(productions_cache: nd.PathLike,
                               attractions_cache: nd.PathLike,
                               ) -> float:
        """Gets the latest modified time from the caches

        Parameters
        ----------
        productions_cache:
            The path to the productions cache to check

        attractions_cache:
            The path to the attractions cache to check

        Returns
        -------
        timestamp:
            A floating point number giving the number of seconds since the
            epoch (see the time module) for the most recent of
            productions_cache and attractions_cache.

        Raises
        ------
        OSError:
            if the file does not exist or is inaccessible.
        """
        productions_time = os.path.getmtime(productions_cache)
        attractions_time = os.path.getmtime(attractions_cache)
        if productions_time > attractions_time:
            return productions_time
        return attractions_time

    def _maybe_convert_upper_pa_to_lower(self,
                                         upper_model_matrix_dir: nd.PathLike,
                                         external_matrix_output_dir: nd.PathLike,
                                         lower_model_vector_report_dir: nd.PathLike,
                                         productions_cache: nd.PathLike,
                                         attractions_cache: nd.PathLike,
                                         overwrite_cache: bool,
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cache wrapper for self._convert_upper_pa_to_lower()

        Checks if files exist in productions_cache and attractions_cache.
        If files do exist, they will be loaded in returned.
        If not, self._convert_upper_pa_to_lower() will be run.

        Parameters
        ----------
        upper_model_matrix_dir:
            The directory containing the upper model's output matrices.

        external_matrix_output_dir:
            The directory to output all of the external demand, in the
            lower zoning system. I.E. All demand in lower_zoning_system,
            that is not also in the lower_running_zones.

        lower_model_vector_report_dir:
            The directory to output standard reports of the vectors generated
            for the lower model. This is all demand in the lower_running_zones,
            and all data that is not in the external_matrix_output_dir.

        productions_cache:
            Path to where the productions should be cached.

        attractions_cache:
            Path to where the attractions should be cached.

        overwrite_cache:
            Whether to overwrite any cache that exists, no matter what.

        Returns
        -------
        productions:
            pandas DataFrame of productions. Will have columns named after:
            self.lower_zoning_system.name, segment_names, and 'productions'

        attractions:
            pandas DataFrame of attractions. Will have columns named after:
            self.lower_zoning_system.name, segment_names, and 'attractions'
        """
        # Init
        p_exists = os.path.isfile(productions_cache)
        a_exists = os.path.isfile(attractions_cache)

        # Return the cache only if it is safe to do so
        if not overwrite_cache and p_exists and a_exists:
            # Get the last modified time of matrices and caches
            matrix_modified_time = self._get_latest_matrix_time(upper_model_matrix_dir)
            cache_modified_time = self._get_latest_cache_time(
                productions_cache=productions_cache,
                attractions_cache=attractions_cache,
            )

            # Only return cache if made after the matrices
            if matrix_modified_time < cache_modified_time:
                return file_ops.read_df(productions_cache), file_ops.read_df(attractions_cache)

        # Generate the vectors
        productions, attractions = self._convert_upper_pa_to_lower(
            upper_model_matrix_dir=upper_model_matrix_dir,
            external_matrix_output_dir=external_matrix_output_dir,
            lower_model_vector_report_dir=lower_model_vector_report_dir,
        )

        # Save into cache
        file_ops.write_df(productions, productions_cache)
        file_ops.write_df(attractions, attractions_cache)

        return productions, attractions

    def read_lower_pa(self,
                      upper_model_matrix_dir: nd.PathLike,
                      external_matrix_output_dir: nd.PathLike,
                      lower_model_vector_report_dir: nd.PathLike,
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Converts Upper matrices into vectors for lower model

        During the process of conversion, the external area (the area
        not being used by the lower model) is converted into the
        lower_zoning_system and written out to
        external_matrix_output_dir.
        A set of standard reports are also generated based off of the
        generated vectors. There reports are written out to
        lower_model_vector_report_dir

        Parameters
        ----------
        upper_model_matrix_dir:
            The directory containing the upper model's output matrices

        external_matrix_output_dir:
            The directory to output all of the external demand, in the
            lower zoning system. I.E. All demand in lower_zoning_system,
            that is not also in the lower_running_zones.

        lower_model_vector_report_dir:
            The directory to output standard reports of the vectors generated
            for the lower model. This is all demand in the lower_running_zones,
            and all data that is not in the external_matrix_output_dir.

        Returns
        -------
        productions:
            pandas DataFrame of productions. Will have columns named after:
            self.lower_zoning_system.name, segment_names, and 'productions'

        attractions:
            pandas DataFrame of attractions. Will have columns named after:
            self.lower_zoning_system.name, segment_names, and 'attractions'
        """
        # If no cache path, just get the vectors
        if self.cache_path is None:
            return self._convert_upper_pa_to_lower(
                upper_model_matrix_dir=upper_model_matrix_dir,
                external_matrix_output_dir=external_matrix_output_dir,
                lower_model_vector_report_dir=lower_model_vector_report_dir,
            )

        # Generate cache_paths
        fname = self._production_base_cache.format(
            trip_origin=self.trip_origin,
            zoning=self.lower_zoning_system.name,
            mode=self.running_mode.value,
            tier='lower',
        )
        productions_cache = os.path.join(self.cache_path, fname)

        fname = self._attraction_base_cache.format(
            trip_origin=self.trip_origin,
            zoning=self.lower_zoning_system.name,
            mode=self.running_mode.value,
            tier='lower',
        )
        attractions_cache = os.path.join(self.cache_path, fname)

        # Try load from the cache
        return self._maybe_convert_upper_pa_to_lower(
            upper_model_matrix_dir=upper_model_matrix_dir,
            external_matrix_output_dir=external_matrix_output_dir,
            lower_model_vector_report_dir=lower_model_vector_report_dir,
            productions_cache=productions_cache,
            attractions_cache=attractions_cache,
            overwrite_cache=self.overwrite_cache,
        )

    def build_distribution_model_init_args(self):
        return {
            'year': self.year,
            'trip_origin': self.trip_origin,
            'running_mode': self.running_mode,
            'running_segmentation': self.running_segmentation,
            'upper_model_zoning': self.upper_zoning_system,
            'upper_running_zones': self.upper_running_zones,
            'lower_model_zoning': self.lower_zoning_system,
            'lower_running_zones': self.lower_running_zones,
        }

    @abc.abstractmethod
    def build_upper_model_arguments(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def build_lower_model_arguments(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def build_pa_to_od_arguments(self) -> Dict[str, Any]:
        pass


class DistributionModelArgumentBuilder(DMArgumentBuilderBase):
    # Costs constants
    _modal_dir_name = 'modal'
    _cost_dir_name = 'costs'
    _cost_base_fname = "{zoning_name}_{cost_type}_costs.csv"

    # PA to OD consts
    _tour_props_dir_name = 'pre_me_tour_proportions'
    _fh_th_factors_dir_name = 'fh_th_factors'

    # CJTW constants
    _cjtw_infill = 1e-7
    _cjtw_dir_name = 'cjtw'
    _cjtw_base_fname = 'cjtw_{zoning_name}.csv'

    # Distributors consts
    _distributors_dir = 'distributors'
    _gravity_model_dir = 'gravity model'
    _calibration_mats_dir = 'calibration zones'

    # Trip Length Distribution constants
    _tld_dir_name = 'trip_length_distributions'
    _tld_dir_name2 = 'demand_imports'

    def __init__(self,
                 import_home: nd.PathLike,
                 year: int,
                 trip_origin: str,
                 productions: nd.DVector,
                 attractions: nd.DVector,
                 running_mode: nd.Mode,
                 running_segmentation: nd.SegmentationLevel,
                 upper_zoning_system: nd.ZoningSystem,
                 upper_running_zones: List[Any],
                 lower_zoning_system: nd.ZoningSystem,
                 lower_running_zones: List[Any],
                 target_tld_version: str,
                 init_params_cols: List[str],
                 upper_model_method: nd.DistributionMethod,
                 upper_model_kwargs: Dict[str, Any],
                 upper_init_params_fname: str,
                 upper_target_tld_dir: str,
                 upper_calibration_areas: Union[Dict[Any, str], str],
                 upper_calibration_zones_fname: Optional[str] = None,
                 upper_calibration_naming: Optional[Dict[Any, str]] = None,
                 lower_model_method: Optional[nd.DistributionMethod] = None,
                 lower_model_kwargs: Optional[Dict[str, Any]] = None,
                 lower_init_params_fname: Optional[str] = None,
                 lower_target_tld_dir: str = None,
                 lower_calibration_areas: Optional[Union[Dict[Any, str], str]] = None,
                 lower_calibration_zones_fname: Optional[str] = None,
                 lower_calibration_naming: Optional[Dict[Any, str]] = None,
                 tour_props_version: Optional[str] = None,
                 tour_props_zoning_name: Optional[str] = None,
                 intrazonal_cost_infill: Optional[float] = None,
                 cache_path: Optional[nd.PathLike] = None,
                 overwrite_cache: Optional[nd.PathLike] = False,
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # Set default values
        if not isinstance(upper_calibration_areas, dict):
            upper_calibration_areas = {1: upper_calibration_areas}

        if not isinstance(lower_calibration_areas, dict):
            lower_calibration_areas = {1: lower_calibration_areas}

        super().__init__(
            year=year,
            trip_origin=trip_origin,
            running_mode=running_mode,
            running_segmentation=running_segmentation,
            upper_zoning_system=upper_zoning_system,
            upper_running_zones=upper_running_zones,
            lower_zoning_system=lower_zoning_system,
            lower_running_zones=lower_running_zones,
            cache_path=cache_path,
            overwrite_cache=overwrite_cache,
        )

        # TODO(BT): Validate segments and zones are the correct types

        # Assign attributes
        self.import_home = import_home
        self.productions = productions
        self.attractions = attractions

        self.running_mode = running_mode
        self.upper_target_tld_dir = upper_target_tld_dir
        self.lower_target_tld_dir = lower_target_tld_dir
        self.target_tld_version = target_tld_version

        self.init_params_cols = init_params_cols

        self.upper_model_method = upper_model_method
        self.upper_model_kwargs = upper_model_kwargs
        self.upper_init_params_fname = upper_init_params_fname
        self.upper_calibration_areas = upper_calibration_areas
        self.upper_calibration_zones_fname = upper_calibration_zones_fname
        self.upper_calibration_naming = upper_calibration_naming

        self.lower_model_method = lower_model_method
        self.lower_model_kwargs = lower_model_kwargs
        self.lower_init_params_fname = lower_init_params_fname
        self.lower_calibration_areas = lower_calibration_areas
        self.lower_calibration_zones_fname = lower_calibration_zones_fname
        self.lower_calibration_naming = lower_calibration_naming

        # Tour proportions params
        self.tour_props_version = tour_props_version
        self.tour_props_zoning_name = tour_props_zoning_name

        self.intrazonal_cost_infill = intrazonal_cost_infill

    @staticmethod
    def _maybe_read_trip_end(trip_end: Union[nd.DVector, nd.PathLike]) -> nd.DVector:
        """Read in a Dvector if path was give"""
        # Just return if it's a DVector
        if isinstance(trip_end, nd.core.data_structures.DVector):
            return trip_end

        if not os.path.exists(trip_end):
            raise ValueError(
                "Expected either a DVector or a path to one. No path exists at "
                "%s" % trip_end
            )

        return nd.DVector.load(trip_end)

    def _read_calibration_zones_matrix(self, fname: str) -> np.ndarray:
        """Reads in the matrix of calibration zones"""
        file_path = os.path.join(
            self.import_home,
            self._distributors_dir,
            self._calibration_mats_dir,
            fname,
        )

        df = file_ops.read_df(file_path, index_col=0, find_similar=True)

        return df

    def _get_cost(self,
                  segment_params: Dict[str, Any],
                  zoning_system: nd.ZoningSystem,
                  ) -> pd.DataFrame:
        """Reads in the cost matrix for this segment"""
        # Generate the path to the segment file
        cost_dir = os.path.join(
            self.import_home,
            self._modal_dir_name,
            self.running_mode.value,
            self._cost_dir_name,
            zoning_system.name,
        )
        fname = self.running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc="%s_cost" % zoning_system.name,
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(cost_dir, fname)

        # Read in the costs and infill
        cost_matrix = nd.read_df(path, find_similar=True, index_col=0)
        if self.intrazonal_cost_infill is not None:
            cost_matrix = cost_utils.iz_infill_costs(
                cost_matrix,
                iz_infill=self.intrazonal_cost_infill,
            )

        return cost_matrix

    def _get_target_cost_distribution(self,
                                      area: str,
                                      segment_params: Dict[str, Any],
                                      target_tld_dir: str,
                                      ) -> pd.DataFrame:
        """Reads in the target cost distribution for this segment"""
        # Generate the path to the cost distribution file
        tcd_dir = os.path.join(
            self.import_home,
            self._tld_dir_name,
            self._tld_dir_name2,
            self.target_tld_version,
            area,
            target_tld_dir,
        )

        fname = self.running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc="tlb",
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(tcd_dir, fname)

        # Convert to expected format
        target_cost_distribution = file_ops.read_df(path)

        rename = {'lower': 'min', 'upper': 'max'}
        target_cost_distribution = target_cost_distribution.rename(columns=rename)
        target_cost_distribution['min'] *= constants.MILES_TO_KM
        target_cost_distribution['max'] *= constants.MILES_TO_KM

        return target_cost_distribution

    def _get_init_params(self,
                         segment_params: Dict[str, Any],
                         init_params_df: pd.DataFrame,
                         ) -> Dict[str, int]:
        """Extracts the init params for this segment from init_params_df"""
        seg_init_params = pd_utils.filter_df(init_params_df, segment_params)

        if len(seg_init_params) > 1:
            seg_name = self.running_segmentation.generate_file_name(segment_params)
            raise ValueError(
                "%s rows found in init_params for segment %s. "
                "Expecting only 1 row."
                % (len(seg_init_params), seg_name)
            )

        # Make sure the columns we need do exist
        seg_init_params = pd_utils.reindex_cols(
            df=seg_init_params,
            columns=self.init_params_cols,
            dataframe_name='init_params',
        )

        return {x: seg_init_params[x].squeeze() for x in self.init_params_cols}

    def _build_gravity_by_segment_kwargs(self, init_params_fname: str):
        """Build the dictionary of kwargs for each segment"""
        # Read in the init_params_df
        path = os.path.join(
            self.import_home,
            self._gravity_model_dir,
            init_params_fname
        )
        init_params_df = file_ops.read_df(path)

        # Generate by segment kwargs
        by_segment_kwargs = dict()
        for segment_params in self.running_segmentation:
            # Get the needed kwargs
            segment_name = self.running_segmentation.get_segment_name(segment_params)
            init_params = self._get_init_params(segment_params, init_params_df)

            # Add to dictionary
            by_segment_kwargs[segment_name] = {
                'init_params': init_params,
            }

        return by_segment_kwargs

    def _build_furness3d_by_segment_kwargs(self):
        """furness3d has no by-segment kwargs!"""
        return dict()

    def _build_by_segment_kwargs(self,
                                 method: nd.DistributionMethod,
                                 init_params_fname: str = None,
                                 ) -> Dict[str, Any]:
        if method == nd.DistributionMethod.GRAVITY:
            by_segment_kwargs = self._build_gravity_by_segment_kwargs(init_params_fname)
        elif method == nd.DistributionMethod.FURNESS3D:
            by_segment_kwargs = self._build_furness3d_by_segment_kwargs()
        else:
            raise NotImplementedError(
                "No function exists to build the by_segment_kwargs for "
                "DistributionMethod %s"
                % method.value
            )
        return by_segment_kwargs

    def _build_cost_matrices(self, zoning_system: nd.ZoningSystem):
        """Build the dictionary of cost matrices for each segment"""
        # Generate by segment kwargs
        cost_matrices = dict()
        desc = "Reading in cost"
        # TODO: NOT KLUDGE
        count = 1
        for segment_params in tqdm.tqdm(self.running_segmentation, desc=desc):
            # Get the needed kwargs
            segment_name = self.running_segmentation.get_segment_name(segment_params)
            if count == 1:
                cost_matrix = self._get_cost(segment_params, zoning_system)
                count += 1

            # Add to dictionary
            cost_matrices[segment_name] = cost_matrix

        return cost_matrices

    def _build_target_cost_distributions(self, area: str, target_tld_dir: str):
        """Build the dictionary of target_cost_distributions for each segment"""
        # Generate by segment kwargs
        target_cost_distributions = dict()
        for segment_params in self.running_segmentation:
            # Get the needed kwargs
            segment_name = self.running_segmentation.get_segment_name(segment_params)
            target_cost_distribution = self._get_target_cost_distribution(
                area=area,
                segment_params=segment_params,
                target_tld_dir=target_tld_dir,
            )

            # Add to dictionary
            target_cost_distributions[segment_name] = target_cost_distribution

        return target_cost_distributions

    def _build_furness3d_further_kwargs(self,
                                        zoning_system: nd.ZoningSystem,
                                        running_zones: List[Any],
                                        ) -> Dict[str, Any]:
        # Build a path to the input
        fname = self._cjtw_base_fname.format(zoning_name=zoning_system.name)
        path = os.path.join(self.import_home, self._cjtw_dir_name, fname)

        # Read and convert to zoning system format
        cjtw = read_cjtw(
            file_path=path,
            zoning_name=zoning_system.name,
            subset=None,
            reduce_to_pa_factors=False,
        )

        # Aggregate mode
        p_col = list(cjtw)[0]
        a_col = list(cjtw)[1]
        cjtw = cjtw[cjtw['mode'] == self.running_mode.get_mode_num()].copy()
        cjtw = cjtw.reindex([p_col, a_col, 'trips'], axis=1)
        cjtw = cjtw.groupby([p_col, a_col]).sum().reset_index()

        # Convert to a wide matrix
        base_matrix = pd_utils.long_to_wide_infill(
            df=cjtw,
            index_col=p_col,
            columns_col=a_col,
            values_col='trips',
            index_vals=running_zones,
            column_vals=running_zones,
            infill=self._cjtw_infill,
        )

        return {'base_matrix': base_matrix.values}

    def _build_further_distributor_kwargs(self,
                                          method: nd.Mode,
                                          zoning_system: nd.ZoningSystem,
                                          running_zones: List[Any],
                                          ) -> Dict[str, Any]:
        if method == nd.DistributionMethod.GRAVITY:
            further_kwargs = dict()
        elif method == nd.DistributionMethod.FURNESS3D:
            further_kwargs = self._build_furness3d_further_kwargs(
                zoning_system=zoning_system,
                running_zones=running_zones,
            )
        else:
            raise NotImplementedError(
                "No function exists to build the further_kwargs for "
                "DistributionMethod %s"
                % method.value
            )

        return further_kwargs

    def build_upper_model_arguments(self) -> Dict[str, Any]:
        # Read and validate trip ends
        productions = self._maybe_read_trip_end(self.productions)
        attractions = self._maybe_read_trip_end(self.attractions)

        # Read in calibration zones data
        if self.upper_calibration_zones_fname is not None:
            calibration_matrix = self._read_calibration_zones_matrix(
                self.upper_calibration_zones_fname,
            )
        else:
            calibration_matrix = pd.DataFrame(
                index=self.upper_zoning_system.unique_zones,
                columns=self.upper_zoning_system.unique_zones,
                data=1,
            )

        calib_area_keys = self.upper_calibration_areas.keys()
        target_cost_distributions = dict.fromkeys(calib_area_keys)
        for area_key, area_name in self.upper_calibration_areas.items():
            area_targets = self._build_target_cost_distributions(
                area=area_name,
                target_tld_dir=self.upper_target_tld_dir,
            )
            target_cost_distributions[area_key] = area_targets

        if self.upper_calibration_naming is None:
            upper_calibration_naming = {x: x for x in calib_area_keys}
        else:
            upper_calibration_naming = self.upper_calibration_naming

        # Build dictionaries of arguments for each segment
        cost_matrices = self._build_cost_matrices(self.upper_zoning_system)
        by_segment_kwargs = self._build_by_segment_kwargs(
            self.upper_model_method,
            self.upper_init_params_fname,
        )

        # Read in any further needed kwargs
        further_dist_args = self._build_further_distributor_kwargs(
            method=self.upper_model_method,
            zoning_system=self.upper_zoning_system,
            running_zones=self.upper_running_zones,
        )

        final_kwargs = self.upper_model_kwargs.copy()
        final_kwargs.update(further_dist_args)
        final_kwargs.update({
            'productions': productions.to_df(),
            'attractions': attractions.to_df(),
            'running_segmentation': self.running_segmentation,
            'cost_matrices': cost_matrices,
            'calibration_matrix': calibration_matrix,
            'target_cost_distributions': target_cost_distributions,
            'calibration_naming': upper_calibration_naming,
            'by_segment_kwargs': by_segment_kwargs,
        })

        return final_kwargs

    def build_lower_model_arguments(self) -> Dict[str, Any]:
        # Read in calibration zones data
        if self.lower_calibration_zones_fname is not None:
            calibration_matrix = self._read_calibration_zones_matrix(
                self.lower_calibration_zones_fname,
            )
        else:
            calibration_matrix = pd.DataFrame(
                index=self.lower_zoning_system.unique_zones,
                columns=self.lower_zoning_system.unique_zones,
                data=1,
            )

        calib_area_keys = self.lower_calibration_areas.keys()
        target_cost_distributions = dict.fromkeys(calib_area_keys)
        for area_key, area_name in self.lower_calibration_areas.items():
            area_targets = self._build_target_cost_distributions(
                area=area_name,
                target_tld_dir=self.lower_target_tld_dir,
            )
            target_cost_distributions[area_key] = area_targets

        if self.lower_calibration_naming is None:
            lower_calibration_naming = {x: x for x in calib_area_keys}
        else:
            lower_calibration_naming = self.lower_calibration_naming

        # Build dictionaries of arguments for each segment
        cost_matrices = self._build_cost_matrices(self.lower_zoning_system)
        by_segment_kwargs = self._build_by_segment_kwargs(
            self.lower_model_method,
            self.lower_init_params_fname,
        )

        # Read in any further needed kwargs
        further_dist_args = self._build_further_distributor_kwargs(
            method=self.lower_model_method,
            zoning_system=self.lower_zoning_system,
            running_zones=self.lower_running_zones,
        )

        final_kwargs = self.lower_model_kwargs.copy()
        final_kwargs.update(further_dist_args)
        final_kwargs.update({
            'running_segmentation': self.running_segmentation,
            'cost_matrices': cost_matrices,
            'calibration_matrix': calibration_matrix,
            'target_cost_distributions': target_cost_distributions,
            'calibration_naming': lower_calibration_naming,
            'by_segment_kwargs': by_segment_kwargs,
        })

        return final_kwargs

    def build_pa_to_od_arguments(self) -> Dict[str, Any]:
        # TODO(BT): UPDATE build_od_from_fh_th_factors() to use segmentation levels
        seg_level = 'tms'
        seg_params = {
            'p_needed': self.running_segmentation.segments['p'].unique(),
            'm_needed': self.running_segmentation.segments['m'].unique(),
        }
        if 'ca' in self.running_segmentation.naming_order:
            seg_params.update({
                'ca_needed': self.running_segmentation.segments['ca'].unique(),
            })

        # Build the factors dir
        fh_th_factors_dir = os.path.join(
            self.import_home,
            self._modal_dir_name,
            self.running_mode.value,
            self._tour_props_dir_name,
            self.tour_props_version,
            self.tour_props_zoning_name,
            self._fh_th_factors_dir_name,
        )

        return {
            'seg_level': seg_level,
            'seg_params': seg_params,
            'fh_th_factors_dir': fh_th_factors_dir,
        }


# ## DEFINE EXPORT PATHS ##
class DistributorExportPaths:
    _reports_dirname = 'Logs & Reports'

    # Output dir names
    _matrix_out_dir = 'Matrices'

    # Report dir names
    _overall_log_name = '{trip_origin}_overall_log.csv'
    _log_dir_name = 'Logs'
    _tld_report_dir = 'TLD Reports'

    def __init__(self,
                 year: int,
                 trip_origin: str,
                 running_mode: nd.Mode,
                 export_home: nd.PathLike,
                 ):
        # Init
        file_ops.check_path_exists(export_home)

        # Assign attributes
        self.year = year
        self.trip_origin = trip_origin
        self.running_mode = running_mode
        self.export_home = export_home
        self.report_home = os.path.join(self.export_home, self._reports_dirname)

        file_ops.create_folder(self.report_home)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> _DistributorExportPaths_NT:
        """Creates self.export_paths"""

        # Build the matrix output path
        matrix_dir = os.path.join(self.export_home, self._matrix_out_dir)

        # Make paths that don't exist
        dir_paths = [matrix_dir]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.export_paths = _DistributorExportPaths_NT(
            home=self.export_home,
            matrix_dir=matrix_dir,
        )

    def _create_report_paths(self) -> _DistributorReportPaths_NT:
        """Creates self.report_paths"""
        # Build paths
        fname = self._overall_log_name.format(trip_origin=self.trip_origin)
        overall_log_path = os.path.join(self.report_home, fname)
        model_log_dir = os.path.join(self.report_home, self._log_dir_name)
        tld_report_dir = os.path.join(self.report_home, self._tld_report_dir)

        # Make paths that don't exist
        dir_paths = [self.report_home, model_log_dir, tld_report_dir]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.report_paths = _DistributorReportPaths_NT(
            home=self.report_home,
            overall_log=overall_log_path,
            model_log_dir=model_log_dir,
            tld_report_dir=tld_report_dir,
        )


class DistributionModelExportPaths:

    # Define the names of the export dirs
    _upper_model_dir = 'Upper Model'
    _lower_model_dir = 'Lower Model'
    _final_outputs_dir = 'Final Outputs'

    # Export dir names
    _upper_external_pa_out_dir = 'Upper External PA Matrices'
    _full_pa_out_dir = 'Full PA Matrices'
    compiled_pa_out_dir = 'Compiled PA Matrices'
    _full_od_out_dir = 'Full OD Matrices'
    _compiled_od_out_dir = 'Compiled OD Matrices'
    _compiled_od_out_dir_pcu = 'PCU'

    # Report dir names
    _reports_dirname = 'Reports'
    _pa_report_dir = 'PA Reports'
    _od_report_dir = 'OD Reports'
    _lower_vector_report_dir = 'Lower Vector Reports'

    def __init__(self,
                 year: int,
                 trip_origin: str,
                 iteration_name: str,
                 running_mode: nd.Mode,
                 upper_model_method: nd.DistributionMethod,
                 lower_model_method: nd.DistributionMethod,
                 export_home: nd.PathLike,
                 ):
        """
        Builds the export paths for all the distribution model

        Parameters
        ----------
        year:
            The year the distribution model is running for

        iteration_name:
            The name of this iteration of the distribution model.
            Will have 'iter' pre-pended to create the folder name. e.g. if
            iteration_name was set to '3i' the iteration folder would be
            called 'iter3i'.

        running_mode:
            The mode that the distribution model is running for. Only one
            mode can be run at a time.

        export_home:
            The home directory of all the export paths. A sub-directory will
            be made for the upper and lower models being run
        """
        # Init
        file_ops.check_path_exists(export_home)

        self.year = year
        self.trip_origin = trip_origin
        self.iteration_name = du.create_iter_name(iteration_name)
        self.running_mode = running_mode
        self.upper_model_method = upper_model_method
        self.lower_model_method = lower_model_method
        self.export_home = os.path.join(export_home, self.iteration_name, self.running_mode.value)
        file_ops.create_folder(self.export_home)

        # ## BUILD ALL MODEL PATHS ## #
        # Upper Model
        self.upper_export_home = os.path.join(self.export_home, self._upper_model_dir)
        file_ops.create_folder(self.upper_export_home)
        self.upper = DistributorExportPaths(
            year=self.year,
            trip_origin=self.trip_origin,
            running_mode=self.running_mode,
            export_home=self.upper_export_home,
        )

        # Lower Model
        self.lower_export_home = os.path.join(self.export_home, self._lower_model_dir)
        file_ops.create_folder(self.lower_export_home)
        self.lower = DistributorExportPaths(
            year=self.year,
            trip_origin=self.trip_origin,
            running_mode=self.running_mode,
            export_home=self.lower_export_home,
        )

        # Final Output Paths
        export_home = os.path.join(self.export_home, self._final_outputs_dir)
        report_home = os.path.join(export_home, self._reports_dirname)
        file_ops.create_folder(export_home)
        file_ops.create_folder(report_home)

        # Generate the paths
        self._create_export_paths(export_home)
        self._create_report_paths(report_home)

    def _create_export_paths(self, export_home: str) -> None:
        # Build the matrix output path
        upper_external_pa = os.path.join(export_home, self._upper_external_pa_out_dir)
        full_pa_dir = os.path.join(export_home, self._full_pa_out_dir)
        compiled_pa_dir = os.path.join(export_home, self.compiled_pa_out_dir)
        full_od_dir = os.path.join(export_home, self._full_od_out_dir)
        compiled_od_dir = os.path.join(export_home, self._compiled_od_out_dir)
        compiled_od_dir_pcu = os.path.join(compiled_od_dir, self._compiled_od_out_dir_pcu)

        # Create the export_paths class
        self.export_paths = _DM_ExportPaths_NT(
            home=export_home,
            upper_external_pa=upper_external_pa,
            full_pa_dir=full_pa_dir,
            compiled_pa_dir=compiled_pa_dir,
            full_od_dir=full_od_dir,
            compiled_od_dir=compiled_od_dir,
            compiled_od_dir_pcu=compiled_od_dir_pcu,
        )

        # Make all paths that don't exist
        dir_paths = [
            upper_external_pa,
            full_pa_dir,
            compiled_pa_dir,
            full_od_dir,
            compiled_od_dir,
            compiled_od_dir_pcu,
        ]
        for path in dir_paths:
            file_ops.create_folder(path)

    def _create_report_paths(self,
                             report_home: str,
                             ) -> None:
        """Creates self.report_paths"""
        # Create the overall report paths
        self.report_paths = _DM_ReportPaths_NT(
            home=report_home,
            pa_reports_dir=os.path.join(report_home, self._pa_report_dir),
            od_reports_dir=os.path.join(report_home, self._od_report_dir),
            lower_vector_reports_dir=os.path.join(report_home, self._lower_vector_report_dir),
        )

        # Make paths that don't exist
        for path in self.report_paths:
            file_ops.create_folder(path)


# ## FUNCTIONS ## #
def read_cjtw(file_path: nd.PathLike,
              zoning_name: str,
              subset=None,
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
    mode_cols = list(method_to_mode.keys())

    for col in mode_cols:
        cjtw = cjtw.rename(columns={col: method_to_mode.get(col)})

    cjtw = cjtw.drop('3_Allcategories_Methodoftraveltowork', axis=1)
    cjtw = cjtw.groupby(cjtw.columns, axis=1).sum()
    cjtw = cjtw.reindex(['1_' + zoning_name + 'Areaofresidence',
                         '2_' + zoning_name + 'Areaofworkplace',
                         '1_walk', '2_cycle', '3_car',
                         '5_bus', '6_rail_ug'], axis=1)

    # Pivot
    cjtw = pd.melt(
        cjtw,
        id_vars=['1_' + zoning_name + 'Areaofresidence', '2_' + zoning_name + 'Areaofworkplace'],
        var_name='mode',
        value_name='trips',
    )
    cjtw['mode'] = cjtw['mode'].str[0]
    cjtw['mode'] = cjtw['mode'].astype(int)

    # Build distribution factors
    hb_totals = cjtw.drop('2_' + zoning_name + 'Areaofworkplace', axis=1)
    hb_totals = hb_totals.groupby(['1_' + zoning_name + 'Areaofresidence', 'mode'])
    hb_totals = hb_totals.sum().reset_index()

    hb_totals = hb_totals.rename(columns={'trips': 'zonal_mode_total_trips'})
    hb_totals = hb_totals.reindex(
        ['1_' + zoning_name + 'Areaofresidence', 'mode', 'zonal_mode_total_trips'],
        axis=1
    )

    cjtw = cjtw.merge(
        hb_totals,
        how='left',
        on=['1_' + zoning_name + 'Areaofresidence', 'mode'],
    )

    # Divide by total trips to get distribution factors
    if reduce_to_pa_factors:
        cjtw['distribution'] = cjtw['trips'] / cjtw['zonal_mode_total_trips']
        cjtw = cjtw.drop(['trips', 'zonal_mode_total_trips'], axis=1)
    else:
        cjtw = cjtw.drop(['zonal_mode_total_trips'], axis=1)

    return cjtw
