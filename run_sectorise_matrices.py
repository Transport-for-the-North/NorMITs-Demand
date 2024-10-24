"""Sectors Growth Analysis."""
# ## IMPORTS ## #
# Standard imports
from pathlib import Path

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from normits_demand.matrices import omx_file
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter
from normits_demand.models.forecasting.edge_replicant import (
    wide_mx_2_long_mx,
    long_mx_2_wide_mx,
)


def add_sectors_to_matrix(mx_df: pd.DataFrame, sectors_df: pd.DataFrame) -> pd.DataFrame:
    """Add sectors to a demand matrix.

    Parameters
    ----------
    mx_df : pd.DataFrame
        demand matrix
    sectors_df : pd.DataFrame
        sectors dataframe

    Returns
    -------
    mx_df : pd.DataFrame
        demand matrix with sectors info

    """
    # merge sectors to matrix on origin
    mx_df = mx_df.merge(
        sectors_df, how="left", left_on=["from_model_zone_id"], right_on=["Zone"]
    )
    # rename columns
    mx_df = mx_df.rename(columns={"SectorCA": "from_sector"})
    # keep needed columns
    mx_df = mx_df[["from_sector", "from_model_zone_id", "to_model_zone_id", "Demand"]]
    # merge sectors to matrix on destination
    mx_df = mx_df.merge(
        sectors_df, how="left", left_on=["to_model_zone_id"], right_on=["Zone"]
    )
    # rename columns
    mx_df = mx_df.rename(columns={"SectorCA": "to_sector"})
    # keep needed columns
    mx_df = mx_df[
        ["from_sector", "from_model_zone_id", "to_model_zone_id", "to_sector", "Demand"]
    ]

    return mx_df


# matrices dictionary
matrices = {
    "ILP_18": Path(
        r"C:\NorMITs\NorTMS_T3_Model_v8.16b\Runs\ILP_2018\Inputs\Demand\PT_24hr_Demand.MAT"
    )
}

sectors_lookup = pd.read_csv(r"C:\NorMITs\GIS\Sqex_TfNZones_to_CASectors.csv", usecols=[0, 1])
cube_exe = Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE")
output_path = Path(r"C:\NorMITs\Analysis\Reporting")
ZONES = 1300


# create sectorized matrices dictionary
sector_mxs = {}

# loop over matrices
for run, mx in matrices.items():
    cm = CUBEMatConverter(cube_exe)
    cm.mat_2_omx(
        f"{mx}",
        output_path,
        f"{run}",
    )
    # create total matrix
    total_mx = np.empty(shape=[ZONES, ZONES])
    # read demand matrix
    with omx_file.OMXFile(Path(output_path / f"{run}.omx")) as omx_mat:
        for mx_lvl in omx_mat.matrix_levels:
            # get matrix level and sum to total demand
            mat = omx_mat.get_matrix_level(mx_lvl)
            total_mx = total_mx + mat
    # move matrix level to a dataframe
    total_mx = wide_mx_2_long_mx(
        total_mx,
        rows="from_model_zone_id",
        cols="to_model_zone_id",
    )
    # add sectors
    total_mx = add_sectors_to_matrix(total_mx, sectors_lookup)
    # group by sectors
    total_mx = total_mx.groupby(["from_sector", "to_sector"])["Demand"].sum().reset_index()
    total_mx = total_mx.loc[
        (total_mx["from_sector"] > 0) & (total_mx["to_sector"] > 0)
    ].reset_index(drop=True)
    # add to dict
    sector_mxs[run] = long_mx_2_wide_mx(total_mx)
    # export to csv
    np.savetxt(output_path / f"{run}.csv", sector_mxs[run], delimiter=",")
