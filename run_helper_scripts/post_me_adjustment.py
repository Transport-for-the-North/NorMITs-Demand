# -*- coding: utf-8 -*-
"""
    Compares demand before and after SATURN's matrix estimation process.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
from pathlib import Path
import shutil
import sys
from typing import NamedTuple, Iterator, Tuple

# Third party imports
import pandas as pd

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import file_ops, general
from normits_demand.matrices import ufm_converter, omx_file, od_to_pa

# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.post_me_adjustment"
)
LOG_FILE = "SATURN_ME_comparison.log"
TIME_PERIODS = {1: "AM", 2: "IP", 3: "PM"}
USERCLASSES = {1: "business", 2: "commute", 3: "other"}
COMPARISON_ZONING = "lad_2020"
COMPARISON_SEGMENTATIONS = {"hb_p_m", "hb_p_m_tp_wday"}

##### CLASSES #####
class SATURNMatrices(NamedTuple):
    """Stores paths to a single SATURN UFM for each time period."""

    am: Path
    ip: Path
    pm: Path


@dataclasses.dataclass
class PostMEAdjustmentParameters:
    """Stores parameters for `post_me_adjustment` module."""

    saturn_folder: Path
    synthetic_full_od_folder: Path
    synthetic_compiled_od_folder: Path
    compile_factors: Path
    post_me_matrices: SATURNMatrices
    output_folder: Path
    model_name: str
    year: int
    mode: nd.Mode = nd.Mode.CAR


class MatrixDetails(NamedTuple):
    """Stores paths to the prior and post ME matrices and the parameters for them."""

    prior_path: Path
    post_path: Path
    parameters: dict[str, str]


##### FUNCTIONS #####
def process_post_me(
    ufm: Path,
    output_folder: Path,
    converter: ufm_converter.UFMConverter,
    time_slice: int,
    year: int,
    mode: int,
    overwrite: bool = False,
) -> None:
    """Extract each UC from Post ME UFM.

    Parameters
    ----------
    ufm : Path
        Path to Post ME UFM.
    output_folder : Path
        Folder to save the UC matrices to.
    converter : ufm_converter.UFMConverter
        Instance of `UFMConverter` to perform the UFM to OMX
        conversion.
    time_slice : int
        Number of the time slice for the given `ufm`, used
        in the output names.
    year : int
        Model year for matrices.
    mode : int
        Mode number for matrices.
    overwrite : bool , default False
        Whether to re-create existing output matrices.
    """
    LOG.info("Extracting Post ME matrices: %s", ufm.name)
    omx_path = converter.ufm_to_omx(ufm)

    # Extract UC 1 - 3 matrices
    with omx_file.OMXFile(omx_path) as omx:
        for uc, nm in USERCLASSES.items():
            out = output_folder / f"od_{nm}_yr{year}_m{mode}_tp{time_slice}.csv.bz2"
            if out.is_file() and not overwrite:
                continue

            lvl_nm = omx.matrix_levels[uc - 1]
            mat = omx.get_matrix_level(lvl_nm)
            mat = pd.DataFrame(mat, index=omx.zones, columns=omx.zones)
            file_ops.write_df(mat, out)
            LOG.info("Written: %s", out)


def copy_prior_tp4(prior_folder: Path, out_folder: Path) -> None:
    """Copy time period 4 prior matrices into output folder.

    ME isn't run on time period 4 but the decompilation process
    needs all 4 time periods to run so copying the priors.

    Parameters
    ----------
    prior_folder : Path
        Folder containing the compiled prior synthetic matrices.
    out_folder : Path
        Folder to copy the matrices to.
    """
    LOG.info("Copying prior matrices from: %s", prior_folder)
    for file in prior_folder.glob("*_tp4.*"):
        out_path = out_folder / file.name.removeprefix("synthetic_")
        LOG.info("Copying %s to %s", file.name, out_path)
        shutil.copy(file, out_path)


def iter_od_matrices(prior_folder: Path, post_folder: Path) -> Iterator[MatrixDetails]:
    """Iterates through the prior matrices and finds the relevant post ME matrix.

    Logs a warning message if a Prior matrix is found without a
    correponding post ME matrix.

    Parameters
    ----------
    prior_folder : Path
        Folder containing prior matrices, files are expected
        to have one of the following suffixes: '.csv',
        '.csv.bz2' or '.pbz2'.
    post_folder : Path
        Folder containing post matrices, post matrices should
        have the same name as the corresponding prior.

    Yields
    ------
    MatrixDetails
        Paths to the prior and post ME matrices and the
        segmentation parameters for them.
    """
    suffixes = (".csv", ".csv.bz2", ".pbz2")
    for prior in prior_folder.iterdir():
        if prior.is_dir() or prior.suffix.lower() not in suffixes:
            continue
        stem = prior.name.removesuffix("".join(prior.suffixes))
        stem = stem.replace("synthetic_", "")

        # Only use HB from home matrices
        params = general.fname_to_calib_params(stem, True, True)
        if params["matrix_format"] != "od_from":
            continue

        for suff in suffixes:
            post = post_folder / (stem + suff)
            if post.is_file():
                yield MatrixDetails(prior, post, params)
                break
        else:
            LOG.warning("Post ME matrix doesn't exist: %s", post)
            continue


def _read_matrix(path: Path) -> pd.DataFrame:
    """Read matrix file and convert columns/indices to integers (if possible)."""
    mat = file_ops.read_df(path, find_similar=True, index_col=0)
    mat.columns = pd.to_numeric(mat.columns, downcast="integer", errors="ignore")
    mat.index = pd.to_numeric(mat.index, downcast="integer", errors="ignore")
    return mat


def production_trip_ends(
    prior_path: Path, post_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = _read_matrix(prior_path)
    post = _read_matrix(post_path)

    if prior.shape != post.shape:
        raise ValueError("prior and post matrices don't have the same shape")
    if (prior.index != post.index).all():
        raise ValueError("prior and post matrices don't have the same indices")
    if (prior.columns != post.columns).all():
        raise ValueError("prior and post matrices don't have the same shape")

    productions = []
    for mat in (prior, post):
        productions.append(mat.sum(axis=1))
    return tuple(productions)


def combined_production_trip_ends(
    prior_folder: Path, post_folder: Path, model_name: str, output_folder: Path
) -> tuple[nd.DVector, nd.DVector]:
    LOG.info("Creating productions DVectors")
    segmentation = nd.get_segmentation_level("hb_p_m_tp_wday")
    zoning = nd.get_zoning_system(model_name)
    prior_trip_ends = {}
    post_trip_ends = {}
    for mat_details in iter_od_matrices(prior_folder, post_folder):
        prior, post = production_trip_ends(mat_details.prior_path, mat_details.post_path)
        seg_name = segmentation.get_segment_name(mat_details.parameters)
        prior_trip_ends[seg_name] = prior.values
        post_trip_ends[seg_name] = post.values
    prior_dvec = nd.DVector(segmentation, prior_trip_ends, zoning, "avg_day")
    post_dvec = nd.DVector(segmentation, post_trip_ends, zoning, "avg_day")

    output_folder.mkdir(exist_ok=True)
    for nm, dvec in (("prior", prior_dvec), ("post", post_dvec)):
        out = output_folder / f"{nm}_productions_{model_name}_{segmentation.name}_dvec.pkl"
        dvec.save(out)
        LOG.info("Written: %s", out)
    return prior_dvec, post_dvec


def compare_productions(prior: nd.DVector, post: nd.DVector, output_path: Path) -> None:
    for seg in COMPARISON_SEGMENTATIONS:
        if seg != prior.segmentation.name:
            segmentation = nd.get_segmentation_level(seg)
            new_prior = prior.aggregate(segmentation)
            new_post = post.aggregate(segmentation)
        else:
            new_prior = prior
            new_post = post
        comp = new_post / new_prior
        out = output_path.with_name(output_path.stem + "-seg.csv.bz2")
        file_ops.write_df(comp.to_df(), out, index=False)
        LOG.info("Written: %s", out)


def main(params: PostMEAdjustmentParameters, init_logger: bool = True) -> None:

    params.output_folder.mkdir(exist_ok=True, parents=True)
    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running Post ME Adjustment",
        )

    # Extract UC matrices from UFMs and convert to HB fh/th and NHB
    post_me_out = params.output_folder / "Post ME"
    post_me_out.mkdir(exist_ok=True)
    converter = ufm_converter.UFMConverter(params.saturn_folder)
    for ts in TIME_PERIODS:
        process_post_me(
            params.post_me_matrices[ts - 1],
            post_me_out,
            converter,
            ts,
            params.year,
            params.mode.get_mode_num(),
        )
    copy_prior_tp4(params.synthetic_compiled_od_folder, post_me_out)

    compiled_out = post_me_out / "Full OD"
    compiled_out.mkdir(exist_ok=True)
    od_to_pa.decompile_od(
        post_me_out,
        compiled_out,
        year=params.year,
        decompile_factors_path=params.compile_factors,
    )

    # Calculate productions trip ends and compare between the two
    compare_out = params.output_folder / "Prior Post Comparisons"
    prior_dvec, post_dvec = combined_production_trip_ends(
        params.synthetic_full_od_folder, compiled_out, params.model_name, compare_out
    )

    out_nm = "prior_post_comparison_productions_{}"
    out_path = compare_out / out_nm.format(params.model_name)
    compare_productions(prior_dvec, post_dvec, out_path)
    comp_zoning = nd.get_zoning_system(COMPARISON_ZONING)
    compare_productions(
        prior_dvec.translate_zoning(comp_zoning),
        prior_dvec.translate_zoning(comp_zoning),
        out_path.with_name(out_nm.format(comp_zoning.name)),
    )


##### MAIN #####
if __name__ == "__main__":
    # Setup parameters, TODO replace this with a config
    # post_me_folder = Path(r"T:\MidMITs Demand\MiHAM Assignments\iter9.3.3\ME")
    post_me_folder = Path(
        r"C:\WSP_Projects\MidMITs\02 MidMITs\Outputs\MiHAM Assignments\iter9.3.3\ME"
    )
    post_me_files = [
        post_me_folder / f"{t}_MELoop6/miham_stacked_TS{n}_I6.UFM"
        for n, t in TIME_PERIODS.items()
    ]
    compiled_od_folder = Path(
        r"T:\MidMITs Demand\Distribution Model\iter9.3.3"
        r"\car_and_passenger\Final Outputs\Compiled OD Matrices"
    )

    parameters = PostMEAdjustmentParameters(
        # Can't use this version of SATURN because it doesn't include the UFM2OMX batch file
        # saturn_folder=Path(r"C:\SATWIN\XEXES11312U"),
        saturn_folder=Path(r"C:\SATWIN\XEXES 11.5.05H MC N4"),
        synthetic_full_od_folder=Path(
            r"T:\MidMITs Demand\Distribution Model\iter9.3.3"
            r"\car_and_passenger\Final Outputs\Full OD Matrices"
        ),
        synthetic_compiled_od_folder=compiled_od_folder,
        compile_factors=compiled_od_folder / "od_compilation_factors.pkl",
        post_me_matrices=SATURNMatrices(*post_me_files),
        output_folder=Path(
            r"C:\WSP_Projects\MidMITs\02 MidMITs\Outputs"
            r"\Post ME Trip Rate Adjustments\iter9.3.3"
        ),
        model_name="miham",
        year=2018,
    )
    main(parameters)
