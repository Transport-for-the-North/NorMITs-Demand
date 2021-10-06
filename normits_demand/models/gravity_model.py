# -*- coding: utf-8 -*-
"""
Created on: 06/10/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by: Chris Storey

File purpose:

"""
# Built-Ins
import os

from typing import Optional

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand.utils import file_ops
from normits_demand.utils import costs as cost_utils
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.validation import checks

from normits_demand.pathing.travel_market_synthesiser import GravityModelExportPaths


class GravityModel(GravityModelExportPaths):
    _log_fname = "Gravity_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 zoning_system: nd.core.ZoningSystem,
                 export_home: nd.PathLike,
                 zone_col: str = None,
                 ):
        # Validate inputs
        if not isinstance(zoning_system, nd.core.zoning.ZoningSystem):
            raise ValueError(
                "Expected and instance of a normits_demand ZoningSystem. "
                "Got a %s instance instead."
                % type(zoning_system)
            )

        # Assign attributes
        self.zoning_system = zoning_system
        self.zone_col = zone_col

        if self.zone_col is None:
            self.zone_col = zoning_system.col_name

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Logs & Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            year=year,
            running_mode=running_mode,
            export_home=export_home,
            report_home=report_home,
        )

        # Create a logger
        logger_name = "%s.%s" % (__name__, self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new Gravity Model Logger",
        )

    def run(self,
            trip_origin: str,
            running_segmentation: nd.core.segments.SegmentationLevel,
            productions: pd.DataFrame,
            attractions: pd.DataFrame,
            init_params: pd.DataFrame,
            costs_path: nd.PathLike,
            pa_val_col: Optional[str] = 'val',
            intrazonal_cost_infill: Optional[float] = 0.5,
            ):


        for segment_params in running_segmentation:
            name = running_segmentation.generate_file_name(segment_params)
            self._logger.info("Running for %s" % name)

            # ## GET P/A VECTORS FOR THIS SEGMENT ## #
            # Figure out which columns we need
            segments = list(segment_params.keys())
            needed_cols = segments + [self._pa_val_col]
            rename_cols = {pa_val_col: self._pa_val_col}

            # Filter productions
            seg_productions = pd_utils.filter_df(productions, segment_params)
            seg_productions = seg_productions.rename(columns=rename_cols)
            seg_productions = seg_productions.set_index(self.zone_col)
            seg_productions = seg_productions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Filter attractions
            seg_attractions = pd_utils.filter_df(attractions, segment_params)
            seg_attractions = seg_attractions.rename(columns=rename_cols)
            seg_attractions = seg_attractions.set_index(self.zone_col)
            seg_attractions = seg_attractions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Check we actually got something
            production_sum = seg_productions.values.sum()
            attraction_sum = seg_attractions.values.sum()
            if production_sum <= 0 or attraction_sum <= 0:
                raise nd.NormitsDemandError(
                    "Missing productions and/or attractions after filtering to "
                    "this segment.\n"
                    "\tSegment: %s\n"
                    "\tProductions sum: %s\n"
                    "\tAttractions sum: %s"
                    % (segment_params, production_sum, attraction_sum)
                )

            # Balance A to P
            adj_factor = seg_productions[self._pa_val_col].sum() / seg_attractions[
                self._pa_val_col].sum()
            seg_attractions[self._pa_val_col] *= adj_factor

            # TODO: MIGHT NEED TO GET P?A AS NUMPY

            # ## GET THE COSTS FOR THIS SEGMENT ## #
            self._logger.debug("Getting costs from: %s" % costs_path)

            int_costs, cost_name = cost_utils.get_costs(
                costs_path,
                segment_params,
                iz_infill=intrazonal_cost_infill,
                replace_nhb_with_hb=(trip_origin == 'nhb'),
            )

            # Translate costs to array
            costs = pd_utils.long_to_wide_infill(
                df=int_costs,
                index_col='p_zone',
                columns_col='a_zone',
                values_col='cost',
                index_vals=self.zoning_system.unique_zones,
                column_vals=self.zoning_system.unique_zones,
                infill=0,
            )

        hb_distribution = gm.run_gravity_model(
            zone_col=zone_col,
            segment_params=calib_params,
            init_param_a=init_param_a,
            init_param_b=init_param_b,
            productions=productions,
            attractions=attractions,
            internal_zones=nup.get_internal_area(self.lookup_folder),
            model_lookup_path=i_paths['lookups'],
            target_tld=target_tld,
            dist_log_path=o_paths['reports'],
            dist_log_fname=trip_origin + '_internal_distribution',
            dist_function=dist_function,
            cost_type=cost_type,
            apply_k_factoring=True,
            furness_loops=furness_loops,
            fitting_loops=fitting_loops,
            bs_con_target=.95,
            target_r_gap=1,
            rounding_factor=3,
            iz_cost_infill=iz_cost_infill,
            verbose=verbose
        )