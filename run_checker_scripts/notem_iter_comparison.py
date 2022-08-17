# -*- coding: utf-8 -*-
"""
    Script to compare two iterations of NoTEM.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
import re
from pathlib import Path
from typing import Dict, Set, Tuple

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('..')

# Local imports
# pylint: disable=import-error
import normits_demand as nd
from normits_demand import logging as nd_log

# pylint: enable=import-error


###### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".notem_iter_comparison")
LOG_FILE = "NoTEM_iter_comparison.log"

BASE_FOLDER = r"F:\NorMITs Demand\NoTEM"
COMPARE_ITERS = ("iter9.10", "iter9.10-tfn")
SCENARIOS = (nd.Scenario.SC01_JAM, nd.Scenario.SC01_JAM)
OUTPUT_FOLDER = r"F:\NorMITs Demand\NoTEM\comparisons"


###### CLASSES #####
@dataclasses.dataclass
class ComparisonInputs:
    """Input parameters for running NoTEM iter comparison script."""

    base_folder: Path
    compare_iters: tuple[str, str]
    scenarios: tuple[nd.Scenario, nd.Scenario]
    output_folder: Path


###### FUNCTIONS #####
def find_and_compare_notem_reports(
    base_folder: Path,
    iters: Tuple[str, str],
    scenarios: tuple[nd.Scenario, nd.Scenario],
    value_cols: Set,
) -> pd.DataFrame:
    """Search for NoTEM reports and compare between iterations.

    Assumes `base_folder` has two sub-folders named `iters` then
    looks through folders to find reports e.g.
    `{base_folder}/{iters[0]}/{scenario}/hb_attractions/reports`.
    These reports are then compared to one another to.

    Note that this function assumes that iters[0] is the newer iteration, and
    iters[1] is the older. Hence the names "new" and "old".

    Parameters
    ----------
    base_folder : Path
        Folder containing iteration sub-folders for NoTEM
        outputs.

    iters : Tuple[str, str]
        Names of the two iterations being compared.

    scenarios : Tuple[nd.Scenario, nd.Scenario]
        Name of the scenarios to compare.

    value_cols : Set
        Column names found inside the reports which should be considered
        as values rather than indices.

    Returns
    -------
    reports_df:
        DataFrame with a row for each different report CSV with
        the following columns:
        - iters[0]: path to the new iteration CSV (iters[0])
        - iters[1]: path to the old iteration CSV
        - `f"{old_iter} exists"`: boolean for whether the old file exists
        - "trip_origin": nhb or hb
        - "report_type": name of report e.g. `pure_demand`
        - "year"
        - "sector": type of zoning or sector system report e.g. `lad_totals`
        - "pa": productions or attractions
        - plus additional comparison columns

    Raises
    ------
    NotADirectoryError
        If either of the `iters` folders don't exist.
    """
    # Assume the 0th item is the newest iter
    new_iter = iters[0]
    old_iter = iters[1]
    old_exists_col = f"{old_iter} exists"

    # Define pattern of the notem report fnames
    notem_report_pattern = re.compile(
        r"^(?P<trip_origin>n?hb)_"
        r"(?P<report_type>\w+)_"
        r"(?P<year>\d+)_"
        r"(?P<sector>\w+)$",
        re.I,
    )

    # Double check the paths actually exist
    folders = {
        new_iter: base_folder / new_iter / scenarios[0].name,
        old_iter: base_folder / old_iter / scenarios[1].name,
    }
    for nm, path in folders.items():
        if not path.is_dir():
            raise NotADirectoryError(f"{nm} iteration folder doesn't exist: {path}")

    # Find all the "new" reports and compare to an "old" report
    files = list()
    pbar = tqdm(
        desc=f"Finding files in {folders[new_iter]}",
        bar_format="{l_bar} {n_fmt} [{elapsed}, {rate_fmt}{postfix}]",
    )
    for trip_origin in ("hb", "nhb"):
        for pa in ("productions", "attractions"):
            report_dir = folders[new_iter] / f"{trip_origin}_{pa}" / "reports"
            for path in report_dir.iterdir():
                test_path = folders[old_iter] / path.relative_to(folders[new_iter])

                match = notem_report_pattern.match(path.stem)
                if match is None:
                    print(f"Unexpected filename: {path.stem}")
                    name_attr = {}
                else:
                    name_attr = match.groupdict()
                name_attr["pa"] = pa

                files.append(
                    {
                        new_iter: path,
                        old_iter: test_path,
                        old_exists_col: test_path.exists(),
                        **name_attr,
                    }
                )
                pbar.update()
    pbar.close()
    files = pd.DataFrame(files)

    # Generate the comparisons
    comparisons = list()
    pbar = tqdm(
        files.loc[files[old_exists_col]].to_dict(orient="index").items(),
        total=np.sum(files[old_exists_col].values),
        desc="Comparing files",
        dynamic_ncols=True,
    )
    for index, data in pbar:
        try:
            data = _read_csvs(
                paths=data,
                read_keys=(new_iter, old_iter),
                value_cols=value_cols,
            )
        except ValueError as err:
            # Note error and continue
            comparisons.append({"index": index, "error": str(err)})
            continue

        if len(data) != 2:
            # TODO(MB): Why exit here?
            continue

        # Compare the dataframes and store results
        compared_dfs = _compare_dataframes(data[new_iter], data[old_iter])
        comparisons.append({"index": index, **compared_dfs})

    if not comparisons:
        return files

    comparisons = pd.DataFrame(comparisons).set_index("index")
    comparisons = pd.concat([files, comparisons], axis=1)

    # If all files are identical, this column won't exist
    if "max_difference" in comparisons:
        return comparisons.sort_values("max_difference", ascending=False)
    return comparisons


def _read_csvs(paths: Dict[str, Path], read_keys: Tuple[str, str], value_cols: Set) -> Dict[str, pd.DataFrame]:
    """Reads new and old CSVs for comparison."""
    data = dict()
    for nm in read_keys:
        df = pd.read_csv(paths[nm])
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty:
            data[nm] = df
            continue
        for c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], downcast="integer", errors="ignore")
        index_cols = [c for c in df.columns if c not in value_cols]
        df = df.set_index(index_cols, verify_integrity=True)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        data[nm] = df.sort_index()
    return data


def _compare_dataframes(new: pd.DataFrame, old: pd.DataFrame) -> Dict:
    """Compares 2 DataFrames and returns various comparison statistics."""
    comparisons = {"identical": new.equals(old)}
    if not comparisons["identical"]:
        try:
            comparisons["identical_columns"] = (new.columns == old.columns).all()
        except ValueError:
            comparisons["identical_columns"] = False
        try:
            comparisons["identical_index"] = (new.index == old.index).all()
        except ValueError:
            comparisons["identical_index"] = False
        comparisons["max_difference"] = np.max(np.abs((new - old).values))
        comparisons["max_%_difference"] = np.max(np.abs((new / old) - 1).values)
    return comparisons


def find_dvectors(new_folder: Path, old_folder: Path) -> pd.DataFrame:
    """Find DVectors in `new_folder` and correponding one in `old_folder`.

    Searches for any files ending with `_dvec.pkl`.

    Parameters
    ----------
    new_folder : Path
        Folder containing DVectors for new iteration.
    old_folder : Path
        Folder containing DVectors for old iteration.

    Returns
    -------
    pd.DataFrame
        DataFrame with a row for each different report CSV with
        the following columns:
        - new: path to the new iteration CSV (iters[0])
        - old: path to the old iteration CSV
        - `old_exists`: boolean for whether the old file exists
        - `trip_origin`: nhb or hb
        - `zone_system`: name of zone system
        - `segmentation`: name of segmentation
        - `year`
    """
    pattern = re.compile(
        r"^(?P<trip_origin>n?hb)_"
        r"(?P<zone_system>[a-z]+)_"
        r"(?P<segmentation>[a-z_]+)_"
        r"(?P<year>\d+)_dvec$",
        re.I,
    )

    files = []
    pbar = tqdm(
        desc=f"Finding DVectors in {new_folder}",
        bar_format="{l_bar} {n_fmt} [{elapsed}, {rate_fmt}{postfix}]",
    )
    for path in new_folder.rglob("*_dvec.pkl"):
        match = pattern.match(path.stem)
        old_path = old_folder / path.relative_to(new_folder)
        if match is None:
            tqdm.write(f"Unexpected filename: {path.stem}")
            name_attr = {}
        else:
            name_attr = match.groupdict()
        name_attr["pa"] = path.parent.stem.split("_")[-1]
        files.append(
            {
                "new": path,
                "old": old_path,
                "old_exists": old_path.exists(),
                **name_attr,
            }
        )
        pbar.update()

    pbar.close()
    return pd.DataFrame(files)


def compare_dvectors(new: Path, old: Path, output_folder: Path) -> Path:
    """Compare 2 DVector files.

    Creates the following comparison files in a new sub-folder:
    - `DVector_comparison_summary.xlsx`: difference between segment totals
    - `Difference.csv`: new - old
    - `Percentage Difference.csv`: (new / old) - 1
    - `New DVector.csv`: new DVector as a CSV
    - `Old DVector.csv`: old DVector as a CSV

    Parameters
    ----------
    new : Path
        Path to the new DVector file.
    old : Path
        Path to the old DVector file.
    output_folder : Path
        Folder to save comparison outputs, a new folder
        will be created with the name `new.stem`.

    Returns
    -------
    Path
        Folder where outputs are saved.
    """

    def diffs(new, old) -> Dict[str, pd.DataFrame]:
        return {
            "Difference": new - old,
            "Percentage Difference": (new / old) - 1,
        }

    grouped = {}
    no_zones = {}
    for nm, p in (("new", new), ("old", old)):
        dvec: nd.DVector = nd.DVector.load(p)
        df = dvec.to_df()
        del dvec

        # Drop g, soc, ns and ca segmentation
        grp_cols = ["msoa_zone_id", "p", "m", "tp"]
        grpd = df[grp_cols + ["val"]].groupby(grp_cols).sum()
        grpd = grpd.unstack("m")
        grpd.columns = grpd.columns.droplevel(0)
        grouped[nm] = grpd
        no_zones[nm] = grpd.droplevel(0).groupby(level=["p", "tp"]).sum()

    out_folder = output_folder / new.parent.name / f"{new.stem} comparisons"
    out_folder.mkdir(exist_ok=True, parents=True)
    out = out_folder / "DVector_comparison_summary.xlsx"
    with pd.ExcelWriter(out) as writer:
        for nm, df in diffs(**no_zones).items():
            df.to_excel(writer, sheet_name=nm)

    csvs = {
        **diffs(**grouped),
        "New DVector": grouped["new"],
        "Old DVector": grouped["old"],
    }
    for nm, df in csvs.items():
        out = out_folder / f"{nm}.csv"
        df.to_csv(out)

    return out_folder


def main(params: ComparisonInputs, init_logger: bool = True) -> None:
    """Compares the reports and DVectors between 2 NoTEM iterations.

    Parameters
    ----------
    params : ComparisonInputs
        Parameters for the comparison.
    init_logger : bool, default True
        Initialise logger with log file in `params.output_folder`.

    Raises
    ------
    NotADirectorError
        If `params.base_folder` doesn't exist.
    """
    name = "{}-{} to {}-{} comparison".format(
        params.compare_iters[0],
        params.scenarios[0].name,
        params.compare_iters[1],
        params.scenarios[1].name,
    )
    output_folder = params.output_folder / name
    output_folder.mkdir(parents=True, exist_ok=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            output_folder / LOG_FILE,
            "Running NoTEM Iteration Comparison",
        )

    LOG.info("Input parameters:\n%s", params)
    if not params.base_folder.is_dir():
        raise NotADirectoryError(f"base folder doesn't exist: {params.base_folder}")
    LOG.info("Base folder: %s", params.base_folder)

    # Find and compare all NoTEM reports CSVs
    file_comparisons = find_and_compare_notem_reports(
        base_folder=params.base_folder,
        iters=params.compare_iters,
        scenarios=params.scenarios,
        value_cols={"val"},
    )

    # file_comparisons = compare_reports(files, params.scenarios, old_exists_col, {"val"})
    out = output_folder / (name + "-reports.xlsx")
    file_comparisons.to_excel(out, index=False)
    LOG.info("Written: %s", out)

    # Find and compare all DVectors
    dvectors = find_dvectors(
        params.base_folder / params.compare_iters[0] / params.scenarios[0].name,
        params.base_folder / params.compare_iters[1] / params.scenarios[1].name,
    )
    out = output_folder / (name + "-DVector_summary.csv")
    dvectors.to_csv(out, index=False)
    LOG.info("Written: %s", out)

    pbar = tqdm(
        dvectors.itertuples(index=False),
        total=len(dvectors),
        desc="Comparing DVectors",
        dynamic_ncols=True,
    )
    for row in pbar:
        if row.old_exists:
            out = compare_dvectors(row.new, row.old, output_folder)
            tqdm.write(f"Outputs saved to: {out}")

    LOG.info("Comparisons saved to: %s", output_folder)


##### MAIN #####
if __name__ == "__main__":
    try:
        main(
            ComparisonInputs(
                base_folder=Path(BASE_FOLDER),
                compare_iters=COMPARE_ITERS,
                scenarios=SCENARIOS,
                output_folder=Path(OUTPUT_FOLDER),
            )
        )
    except Exception:
        LOG.critical("Iteration comparison failed with error:", exc_info=True)
        raise
