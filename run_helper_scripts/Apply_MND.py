import pandas as pd
import sys
import os
import pathlib
import normits_demand as nd

sys.path.append("..")
# constants
MSOA = zoning.get_zoning_system("msoa")
LAD = zoning.get_zoning_system("lad_2020")
HB_P_TP_WEEK = segments.get_segmentation_level("hb_p_tp_week")
M_TP_WEEK = segments.get_segmentation_level("m_tp_week")
HB_P_M_TP_WEEK = segments.get_segmentation_level("hb_p_m_tp_week")
test_path = pathlib.Path(r"C:\Projects\MidMITS\Python\outputs\tests")


def mnd_factors(org_dest: str) -> data_structures.DVector:
    """_summary_
   Reads in a csv of MND data and returns a dataframe of factors from 2019-November 2021
   Args:
       org_dest: either 'origin' or 'destination'
   Returns:
       a dvector of mnd factors at LAD zoning, and 'hb_p_tp_week' segmentation
   """
    df = (
        pd.read_csv(r"C:\Projects\MidMITS\Python\outputs\output\mnd_factors.csv")
        .groupby(["LAD", "tp", "p"])
        .sum()
    )
    df.rename(
        index={"AM": 1, "IP": 2, "PM": 3, "OP": 4, "Saturday": 5, "Sunday": 6}, inplace=True
    )
    df["factor"] = (2 * df[f"{org_dest} Nov_2021"]) / (
        df[f"{org_dest}"] + df[f"{org_dest} Nov_2019"]
    )
    df.reset_index(inplace=True)
    unstacked = df[["LAD", "p", "tp", "factor"]]
    dvec = data_structures.DVector(
        segmentation=HB_P_TP_WEEK,
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
    iters: int,
    dft_vec: nd.data_structures.DVector,
    mnd_vec: nd.data_structures.DVector,
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
    mnd_base = base.aggregate(HB_P_TP_WEEK)
    for i in range(iters):
        dvec_agg = dvec.aggregate(M_TP_WEEK)
        mnd_res = dvec_agg / dft_base
        print("mnd_res created")
        adj_dft = dft_vec / mnd_res
        final_dft = dvec * adj_dft
        print(f"Adjusted to DfT factors {i+1} times.")
        dft_res = final_dft.aggregate(HB_P_TP_WEEK).translate_zoning(
            LAD
        ) / mnd_base.translate_zoning(LAD)
        adj_mnd = (mnd_vec / dft_res).translate_zoning(MSOA, weighting="no_weight")
        dvec = final_dft * adj_mnd
    return dvec


def main():
    trips_19 = data_structures.DVector.load(
        r"C:\Projects\MidMITS\Python\outputs\tests\hb_msoa_notem_segmented_2018_dvec.pkl"
    )
    dft_factors = pd.read_csv(r"C:\Projects\MidMITS\Python\outputs\output\dft_factors.csv")
    dft_dvec = data_structures.DVector(
        segmentation=M_TP_WEEK,
        import_data=dft_factors,
        zoning_system=MSOA,
        zone_col="msoa",
        val_col="factor",
        time_format="avg_week",
    )
    dft_21 = trips_19 * dft_dvec
    mnd = mnd_factors("origin")
    agg_19 = trips_19.aggregate(HB_P_TP_WEEK).translate_zoning(LAD)
    agg_21 = dft_21.aggregate(HB_P_TP_WEEK).translate_zoning(LAD)
    dft_res = agg_21 / agg_19
    adj = (mnd / dft_res).translate_zoning(MSOA, weighting="no_weight")
    final = dft_21 * adj
    print("About to begin looping")
    export = loop(final, trips_19, 10, dft_dvec, mnd)
    return export


if __name__ == "__main__":
    DVEC = main()
    DVEC.save(r"C:\Projects\MidMITS\Python\outputs\output\test_4.pkl")
    DVEC.write_sector_reports(
        os.path.join(test_path, "final_seg.csv"),
        os.path.join(test_path, "ca.csv"),
        os.path.join(test_path, "ie.csv"),
        os.path.join(test_path, "final_lad_2.csv"),
        HB_P_M_TP_WEEK,
    )
