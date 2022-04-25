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

# Third Party

# Local Imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand.utils import general as du
from normits_demand.pathing import NoTEMExportPaths
# pylint: enable=import-error,wrong-import-position

# GLOBAL VARIABLES
# TODO(BT): Make these command line arguments
MODE = nd.Mode.WALK
ZONING_NAME = 'tfgm_pt'
YEAR = 2018
IMPORT_DRIVE = "I:/"
NOTEM_SCENARIO = nd.Scenario.SC01_JAM


NOTEM_ITER = '9.7'
PHI_VERSION = "v9.7"


@dataclasses.dataclass
class IOPaths:
    """Build and store IO paths for script"""
    year: int
    import_drive: pathlib.Path
    phi_version_name: str
    notem_scenario: nd.Scenario
    notem_iteration_name: str

    # Optional arguments - probably don't want changing
    nd_dir_name: str = "NorMITs Demand"
    notem_dir_name: str = "NoTEM"
    phi_dir_name: str = "phi_factors"
    import_dir_name: str = "import"

    def __post_init__(self):
        # Set up useful dir paths
        self.normits_dir = self.import_drive / self.nd_dir_name
        self.import_dir = self.normits_dir / self.import_dir_name

        # Set up needed paths
        self.phi_dir = self.import_dir / self.phi_dir_name / self.phi_version_name
        self.notem_paths = NoTEMExportPaths(
            path_years=[self.year],
            scenario=self.notem_scenario,
            iteration_name=self.notem_iteration_name,
            export_home=self.normits_dir / self.notem_dir_name
        )


@dataclasses.dataclass
class RunningArgs:
    """Define arguments for a run"""
    mode: int
    zoning_system_name: str

    # Optional - unlikely to need changing
    used_tps: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4])

    def __post_init__(self):
        self.zoning_system = nd.get_zoning_system(self.zoning_system_name)


def main():
    """Set up a run, then run"""
    io_paths = IOPaths(
        year=YEAR,
        import_drive=IMPORT_DRIVE,
        phi_version_name=PHI_VERSION,
        notem_iteration_name=NOTEM_ITER,
        notem_scenario=NOTEM_SCENARIO,
    )

    running_args = RunningArgs(mode=MODE.value, zoning_system_name=ZONING_NAME)


if __name__ == '__main__':
    main()
