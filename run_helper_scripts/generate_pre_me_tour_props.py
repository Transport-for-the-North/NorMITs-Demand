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
MODE = nd_core.Mode.TRAIN
ZONING_NAME = "norms"
YEAR = 2018
IMPORT_DRIVE = "I:/"
# EXPORT_HOME = r"E:\temp\tour props"
EXPORT_HOME = r"I:\NorMITs Demand\import\modal"
NOTEM_SCENARIO = nd_core.Scenario.SC01_JAM

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
    time_periods: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # Generated attributes
    zoning_system: nd_core.ZoningSystem = dataclasses.field(init=False)

    def __post_init__(self):
        self.zoning_system = nd_core.get_zoning_system(self.zoning_system_name)


def main():
    """Set up a run, then run"""
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


if __name__ == "__main__":
    main()
