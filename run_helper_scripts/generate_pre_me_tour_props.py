# -*- coding: utf-8 -*-
"""
Created on: 06/04/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import sys
import pathlib
import dataclasses

from typing import List
from typing import Union

# Third Party
import tqdm
import numpy as np
import pandas as pd

import normits_demand
import pickle

# Local Imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import core as nd_core
from normits_demand.pathing import NoTEMExportPaths
from normits_demand.tools import tour_proportions
from normits_demand.utils import file_ops

# pylint: enable=import-error,wrong-import-position

# GLOBAL VARIABLES
# TODO(BT): Make these command line arguments
MODE = nd_core.Mode.CAR
ZONING_NAME = "miham_dev"
MSOA_NAME = "msoa"
NHB_TIME_SPLITS = pathlib.Path(r"I:\Projects\Freight and Economy\02 Opening Night\A13 Land-Use\04 - MITs\TEM-Forecasting\Inputs\NorMITs Demand\NoTEM\nhb_productions\nhb_time_split_v3.0.csv")
AT_LOOKUP = pathlib.Path(r"I:\Projects\Freight and Economy\02 Opening Night\A13 Land-Use\04 - MITs\TEM-Forecasting\Inputs\NorMITs Demand\NoTEM\nhb_productions\New Area Types.csv")
OUTPUT_PATH = pathlib.Path(r"I:\Projects\Freight and Economy\02 Opening Night\A13 Land-Use\04 - MITs\TEM-Forecasting\Inputs\import\miham_dev\pre_me_tour_proportions\post_me_nhb_tp_splitting_factors.pkl")
YEAR = 2023
IMPORT_DRIVE = r"I:\Projects\Freight and Economy\02 Opening Night\A13 Land-Use\04 - MITs\TEM-Forecasting\Inputs"
# EXPORT_HOME = r"E:\temp\tour props"
EXPORT_HOME = r"I:\Projects\Freight and Economy\02 Opening Night\A13 Land-Use\04 - MITs\TEM-Forecasting\Inputs\import\miham_dev"

NOTEM_SCENARIO = nd_core.Scenario.DLOG

NOTEM_ITER = "9.10"
RETURN_HOME_VERSION = "v3.0"


@dataclasses.dataclass
class IOPaths:
    """Build and store IO paths for script"""

    year: int
    mode: nd_core.Mode
    import_drive: Union[pathlib.Path, str]
    export_home: Union[pathlib.Path, str]
    return_home_version_name: str
    notem_scenario: nd_core.Scenario
    notem_iteration_name: str
    zoning_system_name: nd_core.ZoningSystem

    # Optional arguments - probably don't want changing
    nd_dir_name: str = "NorMITs Demand"
    notem_dir_name: str = "NoTEM"
    return_home_dir_name: str = "phi_factors"
    import_dir_name: str = "import"
    return_home_file_name: str = "phi_factors_m{mode}.csv"
    tour_prop_dir_name: str = "pre_me_tour_proportions"
    fh_th_dir_name: str = "fh_th_factors"

    def __post_init__(self):
        # Set up useful dir paths
        self.normits_dir = self.import_drive / self.nd_dir_name
        self.import_dir = self.normits_dir / self.import_dir_name

        # Set up needed paths
        self.return_home_dir = (
            self.import_dir / self.return_home_dir_name / self.return_home_version_name
        )
        notem_paths = NoTEMExportPaths(
            path_years=[self.year],
            scenario=self.notem_scenario,
            iteration_name=self.notem_iteration_name,
            export_home=self.normits_dir / self.notem_dir_name,
            zoning_name= self.zoning_system_name
        )

        proxy = notem_paths.hb_production.export_paths
        self.hb_productions_path = proxy.notem_segmented[self.year]

        # Build the export paths
        iter_name = f"v{self.notem_iteration_name}"
        temp = self.export_home / self.mode.value / self.tour_prop_dir_name
        self.tour_props_export_dir = temp / iter_name / self.zoning_system_name
        self.fh_th_factor_export_dir = self.tour_props_export_dir / self.fh_th_dir_name
        file_ops.create_folder(self.tour_props_export_dir)
        file_ops.create_folder(self.fh_th_factor_export_dir)

    def get_return_home_factor_path(self, mode: nd_core.Mode) -> pathlib.Path:
        """Build the path to the return_home factors for `mode`"""
        fname = self.return_home_file_name.format(mode=mode.get_mode_num())
        return self.return_home_dir / fname


@dataclasses.dataclass
class RunningArgs:
    """Define arguments for a run"""

    mode: nd_core.Mode
    zoning_system_name: str

    # Optional - unlikely to need changing
    time_periods: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4,])

    # Generated attributes
    zoning_system: nd_core.ZoningSystem = dataclasses.field(init=False)

    def __post_init__(self):
        self.zoning_system = nd_core.get_zoning_system(self.zoning_system_name)


def main():
    """Set up a run, then run"""


    create_nhb(NHB_TIME_SPLITS, AT_LOOKUP, ZONING_NAME, MSOA_NAME, OUTPUT_PATH)

    io_paths = IOPaths(
        year=YEAR,
        mode=MODE,
        import_drive=pathlib.Path(IMPORT_DRIVE),
        export_home=pathlib.Path(EXPORT_HOME),
        return_home_version_name=RETURN_HOME_VERSION,
        notem_iteration_name=NOTEM_ITER,
        notem_scenario=NOTEM_SCENARIO,
        zoning_system_name=ZONING_NAME,
    )

    running_args = RunningArgs(mode=MODE, zoning_system_name=ZONING_NAME)

    # ## GENERATE OBJECTS FOR RUN ## #
    # Generate the return_home_factors object
    return_home_factors = tour_proportions.ReturnHomeFactors(
        dataframe=pd.read_csv(io_paths.get_return_home_factor_path(running_args.mode)),
        time_periods=running_args.time_periods,
        purpose_col="purpose_from_home",
        time_fh_col="time_from_home",
        time_th_col="time_to_home",
        factor_col="direction_factor",
    )

    # Generate the tp_splits object
    tp_splits = tour_proportions.TimePeriodSplits(
        mode=running_args.mode,
        time_periods=running_args.time_periods,
        zoning_system=running_args.zoning_system,
        dvec=nd_core.DVector.load(io_paths.hb_productions_path),
    )

    # Continue from here!
    ooo = tour_proportions.PreMeTourProportionsGenerator(
        return_home_factors=return_home_factors,
        tp_splits=tp_splits,
    )

    desc = "Generating Tour Proportions"
    for segment_params in tqdm.tqdm(tp_splits.output_seg, desc=desc):
        fname_kwargs = {"trip_origin": "hb", "year": str(YEAR), "ftype": ".pkl", "segment_params": segment_params}

        fname = tp_splits.output_seg.generate_file_name(file_desc='tour_proportions', **fname_kwargs)
        path = os.path.join(io_paths.tour_props_export_dir, fname)
        ooo.write_tour_proportions(segment_params, path, out_dtype=np.float32)

        fname = tp_splits.output_seg.generate_file_name(file_desc='fh_factors', **fname_kwargs)
        path = os.path.join(io_paths.fh_th_factor_export_dir, fname)
        ooo.write_from_home_factors(segment_params, path)

        fname = tp_splits.output_seg.generate_file_name(file_desc='th_factors', **fname_kwargs)
        path = os.path.join(io_paths.fh_th_factor_export_dir, fname)
        ooo.write_to_home_factors(segment_params, path)

def create_nhb(nhb_time_splits_path:pathlib.Path, at_lookup_path: pathlib.Path, zoning_name: str, msoa_name: str, output_path: pathlib.Path)->None:
    nhb_time_splits = pd.read_csv(nhb_time_splits_path)
    at_lookup = pd.read_csv(at_lookup_path)

    zoning = normits_demand.get_zoning_system(zoning_name)
    msoa = normits_demand.get_zoning_system(msoa_name)
    translation = msoa._get_translation_definition(zoning, weighting="population")

    temp_at_lookup = at_lookup.merge(translation, how = "outer", left_on="msoa_area_code", right_on=f"{msoa_name}_zone_id")
    temp_at_lookup = temp_at_lookup.groupby([f"{zoning_name}_zone_id", "tfn_area_type"])[f"{msoa_name}_to_{zoning_name}"].sum().reset_index()
    temp_at_lookup = temp_at_lookup.sort_values(f"{msoa_name}_to_{zoning_name}")
    zoning_at_lookup = temp_at_lookup.drop_duplicates([f"{zoning_name}_zone_id", "tfn_area_type"], keep="first")
    zoning_at_lookup = zoning_at_lookup[[f"{zoning_name}_zone_id", "tfn_area_type"]]

    zoning_time_splits = zoning_at_lookup.merge(nhb_time_splits, how = "left", left_on="tfn_area_type", right_on ="tfn_at")

    zoning_time_splits = zoning_time_splits.set_index(["nhb_p", "nhb_m", "tp"])[[f"{zoning_name}_zone_id", "split"]]

    p_m = {}

    for p in zoning_time_splits.index.get_level_values("nhb_p").unique():
        for m in zoning_time_splits.index.get_level_values("nhb_m").unique():
            p_m_tp = {}
            for tp in zoning_time_splits.index.get_level_values("tp").unique():
                splits = zoning_time_splits.loc[p,m,tp]
                cartesian_product = pd.merge(splits, splits, how="cross", suffixes=("_from", "_to"))


                # Pivot table to create square matrix
                result = cartesian_product.pivot_table(index=f"{zoning_name}_zone_id_from", columns=f"{zoning_name}_zone_id_to",
                                                        values="split_from", aggfunc="first")

                p_m_tp[f"nhb_pa_yr2023_p{p}_m{m}_tp{tp}.csv"]=result
            p_m[f"nhb_pa_yr2023_p{p}_m{m}.csv"] = p_m_tp

    with open(output_path, 'wb') as file:
        pickle.dump(p_m, file)
    return

if __name__ == "__main__":
    main()
