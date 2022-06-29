# imports
import logging
import os
import pathlib
import sys
from typing import Optional

import pandas as pd

sys.path.append("..")
sys.path.append(".")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand.core.data_structures import DVector

# pylint: enable=import-error,wrong-import-position
"""
Reads in MND data and trip-ends by year, hb or nhb, and production or attraction,
then factors the trip-end dvectors using the MND and DfT data.  It will output
six DVectors per year, hb and nhb productions, then two versions of attractions for
each.  The first attraction is the "raw" output, with suffix 'pre-balancing, and the
second is balanced to productions such that furnessing will converge.  The outputs
ready for distribution.
"""
# constants
MSOA = nd.get_zoning_system("msoa")
LAD = nd.get_zoning_system("lad_2020")
HB_P_TP_WEEK = nd.get_segmentation_level("hb_p_tp_week")
NHB_P_TP_WEEK = nd.get_segmentation_level("nhb_p_tp_week")
M_TP_WEEK = nd.get_segmentation_level("m_tp_week")
HB_P_M_TP_WEEK = nd.get_segmentation_level("hb_p_m_tp_week")
NHB_P_M_TP_WEEK = nd.get_segmentation_level("nhb_p_m_tp_week")
SEGS = {
    "hb": {"p_tp": HB_P_TP_WEEK, "p_m_tp": HB_P_M_TP_WEEK},
    "nhb": {"p_tp": NHB_P_TP_WEEK, "p_m_tp": NHB_P_M_TP_WEEK},
}
PA_LOOKUP = {"origin": "productions", "destination": "attractions"}
FACTORS_FOLDER = pathlib.Path(r"T:\MidMITs Demand\Inputs\MND Adjustment")
TRIP_ENDS_FOLDER = pathlib.Path(r"T:\MidMITs Demand\MiTEM\iter9.7\NTEM")
OUTPUT_FOLDER = pathlib.Path(r"T:\MidMITs Demand\MiTEM\iter9.7-COVID\NTEM")
YEARS = [2021, 2030, 2040]
CONVERGENCE = 10
MAX_ITER = 20


def mnd_factors(org_dest: str, hb_nhb: str) -> nd.DVector:
    """_summary_
    Reads in a csv of MND data and returns a dataframe of factors from 2019-November 2021
    Args:
        org_dest: either 'origin' or 'destination'
    Returns:
        a dvector of mnd factors at LAD zoning, and 'hb_p_tp_week' segmentation
    """
    df = (
        pd.read_csv(os.path.join(FACTORS_FOLDER, r"mnd_factors.csv"))
        .groupby(["LAD", "tp", "p"])
        .sum()
    )
    if hb_nhb == "nhb":
        df.drop(7, level=2, axis=0, inplace=True)
        update = {i: i + 10 for i in df.index.get_level_values("p")}
        df.rename(index=update, level=2, inplace=True)
    df.rename(
        index={"AM": 1, "IP": 2, "PM": 3, "OP": 4, "Saturday": 5, "Sunday": 6}, inplace=True
    )
    df["factor"] = (2 * df[f"{org_dest} Nov_2021"]) / (
        df[f"{org_dest}"] + df[f"{org_dest} Nov_2019"]
    )
    df.reset_index(inplace=True)
    unstacked = df[["LAD", "p", "tp", "factor"]]
    dvec = nd.DVector(
        segmentation=SEGS[hb_nhb]["p_tp"],
        import_data=unstacked,
        zoning_system=LAD,
        zone_col="LAD",
        val_col="factor",
        time_format="avg_week",
    )
    return dvec


def loop(
    factored: nd.data_structures.DVector,
    base: nd.data_structures.DVector,
    dft_vec: nd.data_structures.DVector,
    mnd_vec: nd.data_structures.DVector,
    hb_nhb: str,
):
    """
    Matches the target year numbers to sets of factors iteratively, such that
    the final output will be perfectly matched to mnd_vec, and proportional to
    dft_vec

    Args:
        factored: A target year dvec matched to mnd_vec at full segmentation
        base: Base year dvec at full segmentation
        iters: The number of iterations you want running
        dft_vec: A dvec of DfT factors at m_tp_week segmentation
        mnd_vec: A dvec of mnd factors at hb_p_tp_week segmentation

    Returns:
        _type_: a factored dvec
    """
    dvec = factored
    dft_base = base.aggregate(M_TP_WEEK)
    mnd_base = base.aggregate(SEGS[hb_nhb]["p_tp"])
    i = CONVERGENCE + 1
    j = 1
    while i > CONVERGENCE and j < MAX_ITER:
        dvec_agg = dvec.aggregate(M_TP_WEEK)
        mnd_res = dvec_agg / dft_base
        adj_dft = dft_vec / mnd_res
        final_dft = dvec * adj_dft
        logging.info("Adjusted to DfT.")
        dft_res = final_dft.aggregate(SEGS[hb_nhb]["p_tp"]).translate_zoning(
            LAD
        ) / mnd_base.translate_zoning(LAD)
        adj_mnd = (mnd_vec / dft_res).translate_zoning(MSOA, weighting="no_weight")
        dvec_ss = dvec.aggregate(M_TP_WEEK)
        dvec = final_dft * adj_mnd
        logging.info("Adjusted to MND.")
        i = abs(dvec.aggregate(M_TP_WEEK) - dvec_ss).sum()
        j += 1
        logging.info(f"DVector is {i} trips out from the previous iteration.")
    logging.info("Convergence criteria met, writing DVector to pickle file.")
    return dvec


def dvec_path(
    scenario_folder: pathlib.Path,
    orig_dest: str,
    hb_nhb: str,
    year: int,
    suffix: Optional[str] = None,
) -> pathlib.Path:
    """Creates path to a DVector file.

    Parameters
    ----------
    scenario_folder : pathlib.Path
        Scenario folder containing sub-folders for hb/nhb
        productions and attractions.
    orig_dest : str
        Trip end type, 'origin' or 'destination'
    hb_nhb : str
        Trip type, 'hb' or 'nhb'
    year : int
        Year of DVector
    suffix : Optional[str], optional
        Suffix to add to the end of the file name before
        the extenstion

    Returns
    -------
    pathlib.Path
        Path to a new or existing DVector file.
    """
    pa = PA_LOOKUP[orig_dest]
    if suffix is None:
        filename = f"{hb_nhb}_msoa_notem_segmented_{year}_dvec.pkl"
    else:
        filename = f"{hb_nhb}_msoa_notem_segmented_{year}_dvec-{suffix}.pkl"

    output = scenario_folder / f"{hb_nhb}_{pa}" / filename
    output.parent.mkdir(exist_ok=True, parents=True)
    return output


def main(orig_dest: str, hb_nhb: str, year: int) -> nd.DVector:
    logging.info("Beginning initial factoring.")
    trips_19 = nd.DVector.load(dvec_path(TRIP_ENDS_FOLDER, orig_dest, hb_nhb, year))
    dft_factors = pd.read_csv(os.path.join(FACTORS_FOLDER, r"dft_factors.csv"))
    dft_dvec = nd.DVector(
        segmentation=M_TP_WEEK,
        import_data=dft_factors,
        zoning_system=MSOA,
        zone_col="msoa",
        val_col="factor",
        time_format="avg_week",
    )
    dft_21 = trips_19 * dft_dvec
    mnd = mnd_factors(orig_dest, hb_nhb)
    agg_19 = trips_19.aggregate(SEGS[hb_nhb]["p_tp"]).translate_zoning(LAD)
    agg_21 = dft_21.aggregate(SEGS[hb_nhb]["p_tp"]).translate_zoning(LAD)
    dft_res = agg_21 / agg_19
    adj = (mnd / dft_res).translate_zoning(MSOA, weighting="no_weight")
    final = dft_21 * adj
    logging.info("About to begin looping.")
    export = loop(final, trips_19, dft_dvec, mnd, hb_nhb)
    return export


def balance(production: DVector, attraction: DVector, hb_nhb: str) -> DVector:
    """
    Balances attractions to productions to fix furnessing issues, using DVector.balance_at_segments.

    Args:
        production (DVector): trip productions DVector
        attraction (DVector): trip attractions DVector
        hb_nhb (str): either "hb" or "nhb"
    Returns:
        DVector: The input attraction DVector balanced at segment level to the input production DVector
    """
    mode_balancing_zones = dict.fromkeys((1, 2, 5), nd.get_zoning_system("county"))
    attraction_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level(f"notem_{hb_nhb}_output"),
        nd.get_zoning_system("gor"),
        "m",
        mode_balancing_zones,
    )
    new = attraction.balance_at_segments(production, True, attraction_balance_zoning)
    return new


if __name__ == "__main__":
    log_file = OUTPUT_FOLDER / "Apply_MND.log"
    logging.basicConfig(filename=log_file, level=logging.INFO)
    print(f"Writing log file: {log_file}")
    logging.info(
        "Parameters:\nFactors folder: %s\nTrip ends folder: %s\nOutput folder: %s\nYears: %s",
        FACTORS_FOLDER,
        TRIP_ENDS_FOLDER,
        OUTPUT_FOLDER,
        ", ".join(str(i) for i in YEARS),
    )

    for k in YEARS:
        for j in ["nhb", "hb"]:
            output = {}
            for i in ["origin", "destination"]:
                DVEC = main(i, j, k)
                output[i] = DVEC

                suffix = "-pre_balancing" if i == "destination" else None

                DVEC.save(dvec_path(OUTPUT_FOLDER, i, j, k, suffix=suffix))
            adjusted = balance(output["origin"], output["destination"], j)
            adjusted.save(dvec_path(OUTPUT_FOLDER, "destination", j, k))
        # DVEC.write_sector_reports(
        #     os.path.join(FILE_PATH, "final_seg.csv"),
        #     os.path.join(FILE_PATH, "ca.csv"),
        #     os.path.join(FILE_PATH, "ie.csv"),
        #     os.path.join(FILE_PATH, "final_lad_2.csv"),
        #     HB_P_M_TP_WEEK,
        # )
