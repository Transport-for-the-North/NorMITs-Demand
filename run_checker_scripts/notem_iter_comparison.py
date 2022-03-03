# -*- coding: utf-8 -*-
"""
    Script to compare two iterations of NoTEM.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
import logging
import re
from pathlib import Path
from typing import Dict, Set, Tuple

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
# pylint: disable=import-error
import normits_demand as nd
from normits_demand.utils import file_ops

# pylint: enable=import-error


###### CONSTANTS #####
LOG = logging.getLogger(__name__)


###### CLASSES #####
@dataclasses.dataclass
class ComparisonInputs:
    """Input parameters for running NoTEM iter comparison script."""

    base_folder: Path = Path(r"I:\NorMITs Demand\NoTEM")
    compare_iters: tuple[str, str] = ("iter9.5", "iter9.4")
    scenario: str = "SC01_JAM"
    output_folder: Path = Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Outputs\Attraction Balancing Checks"
    )


###### FUNCTIONS #####
def find_notem_reports(
    base_folder: Path, iters: Tuple[str, str], scenario: str
) -> pd.DataFrame:
    """Search for NoTEM iteration HB/NHB productions and attractions reports.

    Assumes `base_folder` has two sub-folders named `iters` then
    looks through folders to find reports e.g.
    `{base_folder}/{iters[0]}/{scenario}/hb_attractions/reports`.

    Parameters
    ----------
    base_folder : Path
        Folder containing iteration sub-folders for NoTEM
        outputs.
    iters : Tuple[str, str]
        Names of the two iterations being compared.
    scenario : str
        Name of the scenario to compare.

    Returns
    -------
    pd.DataFrame
        DataFrame with a row for each different report CSV with
        the following columns:
        - new: path to the new iteration CSV (iters[0])
        - old: path to the old iteration CSV
        - `old_exists`: boolean for whether the old file exists
        - `trip_origin`: nhb or hb
        - `report_type`: name of report e.g. `pure_demand`
        - `year`
        - `sector`: type of zoning or sector system report e.g. `lad_totals`
        - `pa`: productions or attractions
    """
    pattern = re.compile(
        r"^(?P<trip_origin>n?hb)_"
        r"(?P<report_type>\w+)_"
        r"(?P<year>\d+)_"
        r"(?P<sector>\w+)$",
        re.I,
    )
    new_folder = base_folder / iters[0]
    old_folder = base_folder / iters[1]
    files = []

    pbar = tqdm(
        desc=f"Finding files in {new_folder}",
        bar_format="{l_bar} {n_fmt} [{elapsed}, {rate_fmt}{postfix}]",
    )
    for trip_origin in ("hb", "nhb"):
        for pa in ("productions", "attractions"):
            report_dir = new_folder / scenario / f"{trip_origin}_{pa}" / "reports"
            for path in report_dir.iterdir():
                test_path = old_folder / path.relative_to(new_folder)

                match = pattern.match(path.stem)
                if match is None:
                    print(f"Unexpected filename: {path.stem}")
                    name_attr = {}
                else:
                    name_attr = match.groupdict()
                name_attr["pa"] = pa

                files.append(
                    {
                        "new": path,
                        "old": test_path,
                        "old_exists": test_path.exists(),
                        **name_attr,
                    }
                )
                pbar.update()

    pbar.close()
    return pd.DataFrame(files)


def _read_csvs(paths: Dict[str, Path], value_cols: Set) -> Dict[str, pd.DataFrame]:
    """Reads new and old CSVs for comparison."""
    data = {}
    for nm in ("new", "old"):
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
    comparisons = {}
    comparisons["identical"] = new.equals(old)
    if not comparisons["identical"]:
        comparisons["identical_columns"] = (new.columns == old.columns).all()
        comparisons["identical_index"] = (new.index == old.index).all()
        comparisons["max_difference"] = np.max(np.abs((new - old).values))
        comparisons["max_%_difference"] = np.max(np.abs((new / old) - 1).values)
    return comparisons


def compare_reports(files: pd.DataFrame, value_cols: Set) -> pd.DataFrame:
    """Compare all given NoTEM iteration reports.

    Parameters
    ----------
    files : pd.DataFrame
        Paths to the reports to compare should contain, at least,
        columns new and old with paths to the CSVs reports for
        comparison.
    value_cols : Set
        Column names found inside the reports which should be considered
        as values rather than indices.

    Returns
    -------
    pd.DataFrame
        `file` DataFrame with additional comparison columns added.

    See Also
    --------
    find_notem_reports: for getting the `files` parameter
    """
    comparisons = []
    pbar = tqdm(
        files.loc[files["old_exists"]].itertuples(),
        total=np.sum(files["old_exists"].values),
        desc="Comparing files",
        dynamic_ncols=True,
    )
    for paths in pbar:
        try:
            data = _read_csvs(paths._asdict(), value_cols)
        except ValueError as err:
            comparisons.append({"index": paths.Index, "error": str(err)})
            continue
        if len(data) != 2:
            continue
        comparisons.append({"index": paths.Index, **_compare_dataframes(**data)})

    comparisons = pd.DataFrame(comparisons).set_index("index")
    comparisons = pd.concat([files, comparisons], axis=1)
    return comparisons.sort_values("max_difference", ascending=False)


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


def compare_dvectors(new: Path, old: Path, output_folder: Path) -> None:
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
    """

    def diffs(new, old) -> Dict[str, pd.DataFrame]:
        return {
            "Difference": new - old,
            "Percentage Difference": (new / old) - 1,
        }

    grouped = {}
    no_zones = {}
    for nm, p in (("new", new), ("old", old)):
        print(f"Reading: {nm}")
        dvec: nd.DVector = file_ops.read_pickle(p)
        df = dvec.to_df()
        del dvec

        # Drop g, soc, ns and ca segmentation
        grp_cols = ["msoa_zone_id", "p", "m", "tp"]
        grpd = df[grp_cols + ["val"]].groupby(grp_cols).sum()
        grpd = grpd.unstack("m")
        grpd.columns = grpd.columns.droplevel(0)
        grouped[nm] = grpd
        no_zones[nm] = grpd.droplevel(0).groupby(level=["p", "tp"]).sum()

    # FIXME Distinguish between productions and attractions in output folder name
    # currently the out_folder is named purely with the stem of the DVector file
    # name, which causes overwriting when running for both attractions and productions
    out_folder = output_folder / f"{new.stem} comparisons"
    out_folder.mkdir(exist_ok=True, parents=True)
    out = out_folder / "DVector_comparison_summary.xlsx"
    with pd.ExcelWriter(out) as writer:
        for nm, df in diffs(**no_zones).items():
            df.to_excel(writer, sheet_name=nm)
    print(f"Written: {out}")

    csvs = {
        **diffs(**grouped),
        "New DVector": grouped["new"],
        "Old DVector": grouped["old"],
    }
    for nm, df in csvs.items():
        out = out_folder / f"{nm}.csv"
        df.to_csv(out)
        print(f"Written: {nm}")


def main(params: ComparisonInputs) -> None:
    """Compares the reports and DVectors between 2 NoTEM iterations.

    Parameters
    ----------
    params : ComparisonInputs
        Parameters for the comparison.
    """
    print(
        f"Folder: {params.base_folder}",
        f"Exist: {params.base_folder.exists()}",
        sep="\n",
    )

    # Find and compare all NoTEM reports CSVs
    files = find_notem_reports(
        params.base_folder,
        params.compare_iters,
        params.scenario,
    )
    file_comparisons = compare_reports(files, {"val"})
    file_comparisons.to_excel(
        params.output_folder
        / "{}_to_{}_report_comparison.xlsx".format(*params.compare_iters),
        index=False,
    )

    dvectors = find_dvectors(
        params.base_folder / params.compare_iters[0],
        params.base_folder / params.compare_iters[1],
    )
    # TODO Iterate through dvectors to compare each one separately
    # Compare the first DVectors found
    new, old = dvectors.iloc[0, :2]
    compare_dvectors(new, old, params.output_folder)


##### MAIN #####
if __name__ == "__main__":
    main(ComparisonInputs())
