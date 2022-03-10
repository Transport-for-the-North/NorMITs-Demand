# -*- coding: utf-8 -*-
"""
    Script to add segmentation level to NorMITs Demand package.
"""

##### IMPORTS #####
# Standard imports
import argparse
import sys
from pathlib import Path

import pandas as pd

# Third party imports

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand import constants as nd_consts
from normits_demand.utils import file_ops

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.add_segmentation"
)
LOG_FILE = "Add_segmentation.log"
SEGMENTATION_FOLDER = (
    Path(__file__).parent.parent / "normits_demand/core/definitions/segmentations"
)


##### FUNCTIONS #####
def get_arguments() -> argparse.Namespace:
    """Parses command line arguments for `add_segmentation` module.

    Positional arguments:
    - `name`: name of the new segmentation
    - `unique_segments`: path to CSV containing unique segment names

    Optional arguments
    - `overwrite`: overwrite segmentation if it already exists (default False)

    Returns
    -------
    argparse.Namespace
        Namespace containing values for all arguments.

    Raises
    ------
    FileNotFoundError
        If the path given for `unique_segments` isn't an existing file.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("name", help="name of new segmentation")
    parser.add_argument(
        "unique_segments", type=Path, help="path to CSV containing unique segment names"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite segmentation if it already exists",
    )
    args = parser.parse_args()

    if not args.unique_segments.is_file():
        raise FileNotFoundError(
            f"cannot find unique segments CSV: {args.unique_segments}"
        )
    return args


def add_segment(name: str, unique_segments: Path, overwrite: bool = False) -> None:
    """Add new segmentation folder to `SEGMENTATION_FOLDER`.

    Parameters
    ----------
    name : str
        Name of new segmentation folder.
    unique_segments : Path
        CSV containing list of all unique segmentations, the
        column names are used as the segmentation naming order.
    overwrite : bool, default False
        Overwrite segmentation folder if it already exists.

    Raises
    ------
    FileExistsError
        If the segmentation folder already exists and `overwrite`
        is False.
    """
    seg_folder = SEGMENTATION_FOLDER / name
    if seg_folder.is_dir():
        if overwrite:
            LOG.warning("Overwriting segmentation: %s", name)
        else:
            raise FileExistsError(f"segment '{name}' already exists")
    else:
        LOG.info("Adding segmentation: %s", name)

    seg_folder.mkdir(exist_ok=True)

    df = file_ops.read_df(unique_segments, find_similar=True)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    dups = df.duplicated()
    if dups.sum() > 0:
        before = len(df)
        df = df.loc[~dups]
        LOG.warning("Dropped %s rows containing duplicate segments", before - len(df))

    LOG.info("Found %s unique segments", len(df))
    file_ops.write_df(
        df, seg_folder / f"unique_segments{nd_consts.COMPRESSION_SUFFIX}", index=False
    )

    naming_order = [str(c).strip() for c in df.columns]
    LOG.info("Segment naming: %s", naming_order)
    with open(seg_folder / "naming_order.csv", "wt") as file:
        file.writelines("\n".join(naming_order))


def main(init_logger: bool = True) -> None:
    """Add new segmentation folder to NorMITs demand.

    Parameters
    ----------
    init_logger : bool, default True
        Initialise logger with log file in `SEGMENTATION_FOLDER`.

    Raises
    ------
    NotADirectoryError
        If `SEGMENTATION_FOLDER` doesn't exist.
    """
    args = get_arguments()

    if not SEGMENTATION_FOLDER.is_dir():
        raise NotADirectoryError(
            f"cannot find segmentation folder: {SEGMENTATION_FOLDER}"
        )

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            SEGMENTATION_FOLDER / LOG_FILE,
            "Adding Segmentation",
        )

    add_segment(args.name, args.unique_segments, args.overwrite)


##### MAIN #####
if __name__ == "__main__":
    main()
