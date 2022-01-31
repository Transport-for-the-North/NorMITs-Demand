# -*- coding: utf-8 -*-
"""
    Module containing functionality for providing summary spreadsheets
    for the NTEM forecast outputs.
"""

##### IMPORTS #####
# Standard imports
import re
from pathlib import Path
from typing import Dict, Any, List

# Third party imports
import pandas as pd

# Local imports
from normits_demand import core as nd_core
from normits_demand import logging as nd_log
from normits_demand.models.ntem_forecast import NTEMForecastError, LAD_ZONE_SYSTEM
from normits_demand.models.tempro_trip_ends import TEMProTripEnds
from normits_demand.utils import file_ops

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)
COMPARISON_ZONE_SYSTEM = LAD_ZONE_SYSTEM


##### FUNCTIONS #####
def _filename_contents(filename: str) -> Dict[str, Any]:
    """Extract information from matrix filenames.

    Parameters
    ----------
    filename : str
        Filename to extract information form,
        should not include file suffix.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing matrix segmentation
        information with keys:
        - matrix_type
        - trip_end_type
        - year
        - purpose
        - nide

    Raises
    ------
    NTEMForecastError
        If `filename` isn't in the correct format.
    """
    pat = re.compile(
        "^"
        r"(?P<matrix_type>nhb|hb)"
        r"_(?P<trip_end_type>pa|od)"
        r"_yr(?P<year>\d{4})"
        r"_p(?P<purpose>\d{,2})"
        r"_m(?P<mode>\d)"
        "$",
        re.IGNORECASE,
    )
    match = pat.match(filename)
    if match is None:
        raise NTEMForecastError(
            f"filename ({filename!r}) is not in the correct format"
        )
    data = match.groupdict()
    for key in ("year", "purpose", "mode"):
        data[key] = int(data[key])
    return data


def _matrix_trip_ends(path: Path, trip_end_type: str) -> pd.DataFrame:
    """Calculate trip ends for a matrix file.

    Parameters
    ----------
    path : Path
        Path to matrix file.
    trip_end_type : str, {'pa', 'od'}
        Whether trip ends are productions and attractions
        or origins and destinations.

    Returns
    -------
    pd.DataFrame
        Trip end totals with 3 columns:
        - zone_id
        - trip_end_type
        - trips

    Raises
    ------
    NTEMForecastError
        If `trip_end_type` isn't 'pa' or 'od'.
    """
    trip_end_type = trip_end_type.lower().strip()
    if trip_end_type == "pa":
        te_names = ("productions", "attractions")
    elif trip_end_type == "od":
        te_names = ("origins", "destinations")
    else:
        raise NTEMForecastError(
            f"trip_end_type should be 'pa' or 'od' not {trip_end_type}"
        )
    matrix = file_ops.read_df(path, index_col=0, find_similar=True)
    trip_ends = []
    for i, nm in enumerate(te_names):
        df = matrix.sum(axis=i)
        df.index.name = "zone_id"
        df = df.to_frame(name="trips")
        df.insert(0, "trip_end_type", nm)
        trip_ends.append(df.reset_index())
    trip_ends = pd.concat(trip_ends, axis=0)
    return trip_ends


def _compare_trip_ends(
    matrix_trip_ends: pd.DataFrame,
    tempro_data: TEMProTripEnds,
    matrix_zoning: str,
    year: int,
    trip_end_types: List[str],
) -> pd.DataFrame:
    """Compares `matrix_trip_ends` to `tempro_data`.

    Internal functionality for `pa_matrix_comparison`.
    """
    COLUMNS = ["zone_id", "purpose", "mode", "trips"]
    matrix_zoning = nd_core.get_zoning_system(matrix_zoning)
    comparison_zoning = nd_core.get_zoning_system(COMPARISON_ZONE_SYSTEM)
    for mat_type, seg in (("hb", "hb_p_m_car"), ("nhb", "nhb_p_m_car")):
        seg = nd_core.get_segmentation_level(seg)
        for te_type in trip_end_types:
            mask = (
                (matrix_trip_ends["matrix_type"] == mat_type) &
                (matrix_trip_ends["trip_end_type"] == te_type)
            )
            dvec = nd_core.DVector(
                seg,
                matrix_trip_ends.loc[mask, COLUMNS],
                matrix_zoning,
                time_format="avg_day",
                zone_col=COLUMNS[0],
                val_col=COLUMNS[-1],
                df_naming_conversion={
                    "p": "purpose",
                    "m": "mode"
                },
            )
            dvec = dvec.translate_zoning(comparison_zoning)

            # Check TEMPro DVector
            tempro_dvec = getattr(tempro_data, f"{mat_type}_{te_type}")[year]
            if tempro_dvec.segmentation != dvec.segmentation:
                raise NTEMForecastError(
                    "TEMPro trip ends segmentation should be "
                    f"{dvec.segmentation.name} not {tempro_dvec.segmentation.name}"
                )
            if tempro_dvec.zoning_system != dvec.zoning_system:
                raise NTEMForecastError(
                    "TEMPro trip ends zoning system should be "
                    f"{dvec.zoning_system.name} not {tempro_dvec.zoning_system.name}"
                )

            mat_data = dvec.to_df().rename(columns={"val": "matrix"})
            tempro = tempro_dvec.to_df().rename(columns={"val": "tempro"})
            join_cols = [
                *dvec.segmentation.naming_order,
                f"{dvec.zoning_system.name}_zone_id",
            ]
            combined = mat_data.merge(
                tempro, on=join_cols, how="outer", validate="1:1"
            )
            combined = combined.loc[:, join_cols + ["matrix", "tempro"]]
            combined.insert(0, "trip_end_type", te_type)
            combined.insert(0, "matrix_type", mat_type)
            yield combined


def pa_matrix_comparison(
    pa_folder: Path,
    tempro_data: TEMProTripEnds,
    matrix_zone_system: str,
):
    """Calculate PA matrix trip ends and compare to TEMPro.

    Parameters
    ----------
    pa_folder : Path
        Folder containing PA matrices.
    tempro_data : TEMProTripEnds
        TEMPro trip end data.
    matrix_zone_system : str
        The name of the matrix zone system.
    """
    LOG.info("PA matrix trip ends comparison with TEMPro")
    output_folder = pa_folder / "TEMPro Comparisons"
    output_folder.mkdir(exist_ok=True)
    # Extract information from filenames
    files = []
    file_types = (".pbz2", ".csv")
    for p in pa_folder.iterdir():
        if p.is_dir() or p.suffix.lower() not in file_types:
            continue
        try:
            file_data = _filename_contents(p.stem)
        except NTEMForecastError as err:
            LOG.warn(err)
        file_data["path"] = p
        files.append(file_data)
    files = pd.DataFrame(files)
    files.to_csv(pa_folder / "PA matrices list.csv", index=False)

    # Convert tempro_data to LA zoning and make sure segmentation is (n)hb_p_m
    tempro_data = tempro_data.translate_zoning(COMPARISON_ZONE_SYSTEM)
    # Compare trip ends to tempro for all purposes and years
    for yr in files["year"].unique():
        LOG.info("Getting trip ends for %s", yr)
        trip_ends = []
        for row in files.loc[files["year"] == yr].itertuples(index=False):
            df = _matrix_trip_ends(row.path, row.trip_end_type)
            for c in ("matrix_type", "purpose", "mode"):
                df.loc[:, c] = getattr(row, c)
            trip_ends.append(df)
        trip_ends = pd.concat(trip_ends)
        comparison = _compare_trip_ends(
            trip_ends,
            tempro_data,
            matrix_zone_system,
            yr,
            ("productions", "attractions"),
        )
        comparison = pd.concat(comparison)
        comparison.loc[:, "difference"
                      ] = comparison["tempro"] - comparison["matrix"]
        comparison.loc[:, r"% difference"
                      ] = (comparison["tempro"] / comparison["matrix"]) - 1
        out = output_folder / f"PA_TEMPro_comparisons-{yr}.csv"
        file_ops.write_df(comparison, out, index=False)
        LOG.info("Written: %s", out)
