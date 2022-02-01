# -*- coding: utf-8 -*-
"""
    Module containing functionality for providing summary spreadsheets
    for the NTEM forecast outputs.
"""

##### IMPORTS #####
# Standard imports
import re
from pathlib import Path
from typing import Dict, Any, Tuple

# Third party imports
import pandas as pd

# Local imports
from normits_demand import core as nd_core
from normits_demand import logging as nd_log
from normits_demand.models import ntem_forecast, tempro_trip_ends
from normits_demand.utils import file_ops
from normits_demand import efs_constants as efs_consts

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)
COMPARISON_ZONE_SYSTEM = ntem_forecast.LAD_ZONE_SYSTEM


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
        raise ntem_forecast.NTEMForecastError(
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
        raise ntem_forecast.NTEMForecastError(
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
    trip_ends.loc[:, "zone_id"] = pd.to_numeric(
        trip_ends["zone_id"], downcast="integer"
    )
    return trip_ends


def _compare_trip_ends(
    base_trip_ends: Dict[str, Dict[str, nd_core.DVector]],
    forecast_trip_ends: Dict[str, Dict[str, nd_core.DVector]],
    tempro_data: tempro_trip_ends.TEMProTripEnds,
    years: Tuple[int, int],
) -> pd.DataFrame:
    """Compares matrix growth to `tempro_data`.

    Internal functionality for `pa_matrix_comparison`.
    """
    tempro_comparison = []
    for mat_type in base_trip_ends:
        if mat_type not in forecast_trip_ends:
            raise KeyError(f"{mat_type!r} key not in forecast_trip_ends")
        for te_type in base_trip_ends[mat_type]:
            if te_type not in forecast_trip_ends[mat_type]:
                raise KeyError(
                    f"{te_type!r} not in forecast_trip_ends[{mat_type!r}]"
                )
            dvectors = {
                f"matrix_{years[0]}": base_trip_ends[mat_type][te_type],
                f"matrix_{years[1]}": forecast_trip_ends[mat_type][te_type],
            }
            # Check TEMPro DVector
            for y in years:
                dvec = getattr(tempro_data, f"{mat_type}_{te_type}")[y]
                base_dvec = dvectors[f"matrix_{years[0]}"]
                if dvec.segmentation != base_dvec.segmentation:
                    raise ntem_forecast.NTEMForecastError(
                        "TEMPro trip ends segmentation should be "
                        f"{base_dvec.segmentation.name} not "
                        f"{dvec.segmentation.name}"
                    )
                if dvec.zoning_system != base_dvec.zoning_system:
                    raise ntem_forecast.NTEMForecastError(
                        "TEMPro trip ends zoning system should be "
                        f"{base_dvec.zoning_system.name} not "
                        f"{dvec.zoning_system.name}"
                    )
                dvectors[f"tempro_{y}"] = dvec

            # Convert to DataFrames
            dataframes = []
            index_cols = ["p", "m", f"{COMPARISON_ZONE_SYSTEM}_zone_id"]
            for nm, dvec in dvectors.items():
                df = dvec.to_df().rename(columns={"val": nm})
                df = df.set_index(index_cols)
                dataframes.append(df)
            df = pd.concat(dataframes, axis=1)
            df = df.reset_index()
            df.loc[:, "matrix_type"] = mat_type
            df.loc[:, "trip_end_type"] = te_type
            tempro_comparison.append(
                df.set_index(["matrix_type", "trip_end_type"] + index_cols)
            )
    tempro_comparison = pd.concat(tempro_comparison)
    # Calculate growth differences
    tempro_comparison.loc[:, "matrix_growth"] = (
        tempro_comparison[f"matrix_{years[1]}"] /
        tempro_comparison[f"matrix_{years[0]}"]
    )
    tempro_comparison.loc[:, "tempro_growth"] = (
        tempro_comparison[f"tempro_{years[1]}"] /
        tempro_comparison[f"tempro_{years[0]}"]
    )
    tempro_comparison.loc[:, "growth_difference"] = tempro_comparison[
        "matrix_growth"] - tempro_comparison["tempro_growth"]
    tempro_comparison.loc[:, "growth_%_diff"] = (
        tempro_comparison["matrix_growth"] / tempro_comparison["tempro_growth"]
    ) - 1
    return tempro_comparison


def matrix_dvectors(
    matrices: Dict[int, Path],
    segmentation: str,
    trip_end_type: str,
    matrix_zoning: str,
    mode: int,
) -> Dict[str, nd_core.DVector]:
    """Calculate matrix trip ends and convert to DVectors

    Parameters
    ----------
    matrices : Dict[int, Path]
        Paths to the matrices by purpose (keys), the files
        should be either '.csv' or '.pbz2'
    segmentation : str
        Name of the segmentation level for the
        returned DVectors.
    trip_end_type : str, {'pa', 'od'}
        The type of matrices being used.
    matrix_zoning : str
        Name of the zone system of the matrix.
    mode : int
        The mode number of the matrix.

    Returns
    -------
    Dict[str, nd_core.DVector]
        DVectors for both trip ends (productions and attractions
        or origins and destinations). DVectors have the segmentation
        given and the zone system defined by `COMPARISON_ZONE_SYSTEM`.
    """
    # Get matrix trip ends
    trip_ends = []
    for p, path in matrices.items():
        df = _matrix_trip_ends(path, trip_end_type)
        df.loc[:, "p"] = p
        df.loc[:, "m"] = mode
        trip_ends.append(df)
    trip_ends = pd.concat(trip_ends)

    matrix_zoning = nd_core.get_zoning_system(matrix_zoning)
    comparison_zoning = nd_core.get_zoning_system(COMPARISON_ZONE_SYSTEM)
    dvectors = {}
    columns = ["zone_id", "trips", "p", "m"]
    for te_type in trip_ends.trip_end_type.unique():
        # Convert trip ends to DVector
        dvec = nd_core.DVector(
            nd_core.get_segmentation_level(segmentation),
            trip_ends.loc[trip_ends.trip_end_type == te_type, columns],
            matrix_zoning,
            time_format="avg_day",
            zone_col="zone_id",
            val_col="trips",
            df_naming_conversion={
                "p": "p",
                "m": "m"
            },
        )
        dvectors[te_type] = dvec.translate_zoning(comparison_zoning)
    return dvectors


def pa_matrix_comparison(
    ntem_imports: ntem_forecast.NTEMImportMatrices,
    pa_folder: Path,
    tempro_data: tempro_trip_ends.TEMProTripEnds,
):
    """Calculate PA matrix trip ends and compare to TEMPro.

    Parameters
    ----------
    ntem_imports : ntem_forecast.NTEMImportMatrices
        Paths to the input Post ME base matrices.
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
        except ntem_forecast.NTEMForecastError as err:
            LOG.warning(err)
            continue
        file_data["path"] = p
        files.append(file_data)
    files = pd.DataFrame(files)
    files.to_csv(pa_folder / "PA matrices list.csv", index=False)
    index_cols = ["matrix_type", "year", "mode", "purpose"]
    files = files.loc[:, index_cols + ["path"]].set_index(index_cols)

    # Read base matrices
    if ntem_imports.mode != 3:
        raise NotImplementedError("PA matrix comparison only works for mode 3")
    SEGMENTATION = {"hb": "hb_p_m_car", "nhb": "nhb_p_m_car"}
    base_trip_ends = {}
    for nm, seg in SEGMENTATION.items():
        LOG.info("Getting trip ends for base %s", nm.upper())
        base_trip_ends[nm] = matrix_dvectors(
            getattr(ntem_imports, f"{nm}_paths"),
            seg,
            "pa",
            ntem_imports.model_name,
            ntem_imports.mode,
        )

    # Convert tempro_data to LA zoning and make sure segmentation is (n)hb_p_m
    tempro_data = tempro_data.translate_zoning(COMPARISON_ZONE_SYSTEM)
    # Compare trip ends to tempro for all purposes and years
    for yr in files.index.get_level_values("year").unique():
        forecast_trip_ends = {}
        for nm, seg in SEGMENTATION.items():
            LOG.info("Getting trip ends for %s %s", yr, nm.upper())
            indices = pd.IndexSlice[nm, yr, ntem_imports.mode]
            forecast_trip_ends[nm] = matrix_dvectors(
                files.loc[indices, "path"].to_dict(),
                seg,
                "pa",
                ntem_imports.model_name,
                ntem_imports.mode,
            )
        comparison = _compare_trip_ends(
            base_trip_ends,
            forecast_trip_ends,
            tempro_data,
            (efs_consts.BASE_YEAR, yr),
        )
        out = output_folder / f"PA_TEMPro_comparisons-{yr}.csv"
        file_ops.write_df(comparison, out)
        LOG.info("Written: %s", out)
