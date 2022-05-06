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
import sys
import pathlib
import dataclasses

from typing import List

# Third Party
import pandas as pd

# Local Imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import core as nd_core
from normits_demand.pathing import NoTEMExportPaths
from normits_demand.tools import tour_proportions
# pylint: enable=import-error,wrong-import-position

# GLOBAL VARIABLES
# TODO(BT): Make these command line arguments
MODE = nd_core.Mode.WALK
ZONING_NAME = 'tfgm_pt'
YEAR = 2018
IMPORT_DRIVE = "I:/"
NOTEM_SCENARIO = nd_core.Scenario.SC01_JAM


NOTEM_ITER = '9.10'
PHI_VERSION = "v3.0"


@dataclasses.dataclass
class IOPaths:
    """Build and store IO paths for script"""
    year: int
    import_drive: pathlib.Path
    phi_version_name: str
    notem_scenario: nd_core.Scenario
    notem_iteration_name: str

    # Optional arguments - probably don't want changing
    nd_dir_name: str = "NorMITs Demand"
    notem_dir_name: str = "NoTEM"
    phi_dir_name: str = "phi_factors"
    import_dir_name: str = "import"
    phi_file_name: str = "phi_factors_m{mode}.csv"

    def __post_init__(self):
        # Set up useful dir paths
        self.normits_dir = self.import_drive / self.nd_dir_name
        self.import_dir = self.normits_dir / self.import_dir_name

        # Set up needed paths
        self.phi_dir = self.import_dir / self.phi_dir_name / self.phi_version_name
        notem_paths = NoTEMExportPaths(
            path_years=[self.year],
            scenario=self.notem_scenario,
            iteration_name=self.notem_iteration_name,
            export_home=self.normits_dir / self.notem_dir_name
        )

        proxy = notem_paths.hb_production.export_paths
        self.hb_productions_path = proxy.notem_segmented[self.year]

    def get_phi_factor_path(self, mode: nd_core.Mode) -> pathlib.Path:
        """Build the path to the phi factors for `mode`"""
        fname = self.phi_file_name.format(mode=mode.get_mode_num())
        return self.phi_dir / fname


@dataclasses.dataclass
class RunningArgs:
    """Define arguments for a run"""
    mode: nd_core.Mode
    zoning_system_name: str

    # Optional - unlikely to need changing
    used_tps: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # Generated attributes
    zoning_system: nd_core.ZoningSystem = dataclasses.field(init=False)

    def __post_init__(self):
        self.zoning_system = nd_core.get_zoning_system(self.zoning_system_name)


def main():
    """Set up a run, then run"""
    io_paths = IOPaths(
        year=YEAR,
        import_drive=pathlib.Path(IMPORT_DRIVE),
        phi_version_name=PHI_VERSION,
        notem_iteration_name=NOTEM_ITER,
        notem_scenario=NOTEM_SCENARIO,
    )

    running_args = RunningArgs(mode=MODE, zoning_system_name=ZONING_NAME)

    # ## GENERATE OBJECTS FOR RUN ## #
    # Generate the phi factors object
    phi_factors = tour_proportions.PhiFactors(
        dataframe=pd.read_csv(io_paths.get_phi_factor_path(running_args.mode)),
        time_fh_col="time_from_home",
        time_th_col="time_to_home",
        factor_col="direction_factor",
    )

    # Generate the tp_splits object
    tp_splits = tour_proportions.TimePeriodSplits(
        mode=running_args.mode,
        zoning_system=running_args.zoning_system,
        dvec=nd_core.DVector.load(io_paths.hb_productions_path)
    )

    print(phi_factors.get_factors(time_periods=running_args.used_tps))
    print(tp_splits.df)

    # Continue from here!
    # Also, rename phi factors to return_home_factors everywhere!!!
    tour_proportions.PreMeTourProportionsGenerator(
        phi_factors=phi_factors,
        tp_splits=tp_splits,
    )


if __name__ == '__main__':
    main()
