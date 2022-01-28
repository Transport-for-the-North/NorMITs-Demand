# -*- coding: utf-8 -*-
"""
    Module for producing forecast demand constrained to NTEM.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
import re
from pathlib import Path
from typing import Dict, Any, List, Union

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from normits_demand.utils import file_ops, vehicle_occupancy
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand import efs_constants as efs_consts
from normits_demand.distribution import furness
from normits_demand.models.tempro_trip_ends import NTEMForecastError, TEMProTripEnds
from normits_demand.matrices import pa_to_od, matrix_processing

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)
LAD_ZONE_SYSTEM = "lad_2020"
COMPARISON_ZONE_SYSTEM = LAD_ZONE_SYSTEM


##### CLASSES #####
class NTEMImportMatrices:
    """Generates paths to base PostME matrices.

    These matrices are used as the base for the NTEM forecasting.

    Parameters
    ----------
    import_folder : Path
        Path to the import folder, should contain
        the matrices in a sub-path (`MATRIX_FOLDER`).
    year : int
        Model base year.
    model_name : str
        Name of the model to get inputs from, currently
        only works with 'noham'.

    Raises
    ------
    NotImplementedError
        This class only handles the noham model and
        one mode per model.
    """

    MATRIX_FOLDER = "{name}/post_me/tms_seg_pa"
    SEGMENTATION = {"hb": "hb_p_m", "nhb": "nhb_p_m"}

    def __init__(self, import_folder: Path, year: int, model_name: str) -> None:
        file_ops.check_path_exists(import_folder)
        self.year = int(year)
        self.model_name = model_name.lower().strip()
        if self.model_name != "noham":
            raise NotImplementedError(
                "this class currently only works for 'noham' model"
            )
        self.matrix_folder = import_folder / self.MATRIX_FOLDER.format(
            name=self.model_name
        )
        file_ops.check_path_exists(self.matrix_folder)
        self.mode = efs_consts.MODEL_MODES[self.model_name]
        if len(self.mode) == 1:
            self.mode = self.mode[0]
        else:
            raise NotImplementedError(
                "cannot handle models with more than one mode, "
                f"this model ({self.model_name}) has {len(self.mode)} modes"
            )
        self.segmentation = {
            k: nd_core.get_segmentation_level(s)
            for k, s in self.SEGMENTATION.items()
        }
        self._hb_paths = None
        self._nhb_paths = None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(matrix_folder={self.matrix_folder}, "
            f"year={self.year}, model_name={self.model_name})"
        )

    def __repr__(self) -> str:
        return f"{self.__module__}.{self!s}"

    def _check_path(self, hb: str, segment_params: Dict[str, Any]) -> Path:
        """Generate the segment filename and check it exists.

        Parameters
        ----------
        hb : str {'hb', 'nhb'}
            Whether using home-based (hb) or non-home-based (nhb).
        segment_params : Dict[str, Any]
            A dictionary of {segment_name: segment_value}, passed
            to `SegmentationLevel.generate_file_name`.

        Returns
        -------
        Path
            Path to segment matrix file.
        """
        name = self.segmentation[hb].generate_file_name(
            segment_params,
            file_desc="pa",
            trip_origin=hb,
            year=self.year,
        )
        path = self.matrix_folder / name
        file_ops.check_file_exists(path, find_similar=True)
        return path

    def _get_paths(self, hb: str) -> Dict[int, Path]:
        """Get paths to all hb, or nhb, matrices.

        Parameters
        ----------
        hb : str {'hb', 'nhb'}
            Whether using home-based (hb) or non-home-based (nhb).

        Returns
        -------
        Dict[int, Path]
            Dictionary containing paths to matrices (values)
            for each purpose (keys).
        """
        paths = {}
        for seg in self.segmentation[hb]:
            if seg["m"] != self.mode:
                continue
            paths[seg["p"]] = self._check_path(hb, seg)
        return paths

    @property
    def hb_paths(self) -> Dict[int, Path]:
        """Dict[int, Path]
            Paths to home-based matrices for each
            purpose (keys) for the given year.

        See Also
        --------
        normits_demand.constants.ALL_HB_P:
            for a list of all home-based purposes
        """
        if self._hb_paths is None:
            self._hb_paths = self._get_paths("hb")
        return self._hb_paths.copy()

    @property
    def nhb_paths(self) -> Dict[int, Path]:
        """Dict[int, Path]
            Paths to non-home-based matrices for each
            purpose (keys) for the given year.

        See Also
        --------
        normits_demand.constants.ALL_NHB_P:
            for a list of all non-home-based purposes
        """
        if self._nhb_paths is None:
            self._nhb_paths = self._get_paths("nhb")
        return self._nhb_paths.copy()

    def output_filename(
        self,
        trip_origin: str,
        purpose: int,
        year: int,
        compressed: bool = True,
        **kwargs,
    ) -> str:
        """Generate filename for output matrix.

        Parameters
        ----------
        trip_origin : str {'hb', 'nhb'}
            Whether using home-based (hb) or non-home-based (nhb).
        purpose : int
            Purpose number.
        year : int
            The year for the output matrix
        compressed: bool, default True
            Whether the return should be a compressed filetype or not.
        kwargs: keyword arguments, optional
            All other keyword arguments passed to
            `SegmentationLevel.generate_file_name`.

        Returns
        -------
        str
            Name of the output CSV file.
        """
        try:
            seg = self.segmentation[trip_origin]
        except KeyError as err:
            raise NTEMForecastError(
                "hb should be one of %s not %r" %
                (tuple(self.segmentation.keys()), trip_origin)
            ) from err
        return seg.generate_file_name(
            {
                "p": purpose,
                "m": self.mode
            },
            file_desc="pa",
            trip_origin=trip_origin,
            year=year,
            compressed=compressed,
            **kwargs,
        )


##### FUNCTIONS #####
def trip_end_growth(
    tempro_vectors: Dict[int, nd_core.DVector],
    model_zone_system: str,
    zone_weighting: str,
) -> Dict[int, nd_core.DVector]:
    """Calculate growth at LAD level and return it a `model_zone_system`.

    The trip ends are translated to `LAD_ZONE_SYSTEM` to calculate
    growth factors for the internal area and at `model_zone_system`
    for calculating factors in the external area. The returned DVectors
    are in the `model_zone_system`.

    Parameters
    ----------
    tempro_vectors : Dict[int, nd_core.DVector]
        Trip end vectors from TEMPro for all study years,
        keys should be years and must include
        `normits_demand.efs_constants.BASE_YEAR`.
    model_zone_system : str
        Name of the zone system to convert the
        TEMPro growth factors to.
    zone_weighting : str
        Name of the weighting to use when translating
        to `model_zone_system`.

    Returns
    -------
    Dict[int, nd_core.DVector]
        Trip end growth factors in `model_zone_system` as
        `tempro_vectors` base year, contains all years
        from `tempro_vectors` except the base year.

    Raises
    ------
    NTEMForecastError
        If `normits_demand.efs_constants.BASE_YEAR` is not
        in `tempro_vectors`.
    """
    if efs_consts.BASE_YEAR not in tempro_vectors:
        raise NTEMForecastError(
            f"base year ({efs_consts.BASE_YEAR}) data not given"
        )
    growth_zone = nd_core.get_zoning_system(LAD_ZONE_SYSTEM)
    model_zoning = nd_core.get_zoning_system(model_zone_system)
    # Split data into internal and external DVectors
    # for different growth calculations
    base = tempro_vectors[efs_consts.BASE_YEAR]
    base_data = {
        "internal":
            base.translate_zoning(growth_zone),
        "external":
            base.translate_zoning(model_zoning, weighting=zone_weighting),
    }
    masks = {
        "internal":
            np.isin(model_zoning.unique_zones, model_zoning.internal_zones),
        "external":
            np.isin(model_zoning.unique_zones, model_zoning.external_zones),
    }

    growth = {}
    # Function to set the segment values to 0 wherever mask is False
    set_zero = lambda seg, m: np.where(m, seg, 0)
    # Ignore divide by zero warnings and fill with zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        for yr, data in tempro_vectors.items():
            if yr == efs_consts.BASE_YEAR:
                continue
            forecast = {}
            for area, base in base_data.items():
                forecast[area] = data.translate_zoning(
                    growth_zone if area == "internal" else model_zoning,
                    None if area == "internal" else zone_weighting,
                )
                forecast[area] = forecast[area] / base
                # Set any nan or inf values created by dividing by 0 to 0 growth
                forecast[area] = forecast[area].segment_apply(
                    np.nan_to_num, nan=0.0, posinf=0.0, neginf=0.0
                )
                if area == "internal":
                    forecast[area] = forecast[area].translate_zoning(
                        model_zoning, weighting="average"
                    )
                # Set all zones not in the area to 0
                forecast[area] = forecast[area].segment_apply(
                    set_zero, masks[area]
                )
            # Add the internal and external areas back together
            growth[yr] = forecast["internal"] + forecast["external"]
    return growth


def tempro_growth(
    tempro_data: TEMProTripEnds,
    model_zone_system: str,
) -> TEMProTripEnds:
    """Calculate LAD growth factors and return at original zone system.

    Growth factors are calculated at LAD level but are
    returned at `model_zone_system`.

    Parameters
    ----------
    tempro_data : TEMProTripEnds
        TEMPro trip end data for all study years.
    model_zone_system : str
        Name of the zone system to convert the
        TEMPro growth factors to.

    Returns
    -------
    TEMProTripEnds
        TEMPro trip end growth factors for
        all future years.

    Raises
    ------
    NTEMForecastError
        If `tempro_data` isn't an instance of `TEMProTripEnds`.
    """
    if not isinstance(tempro_data, TEMProTripEnds):
        raise NTEMForecastError(
            f"tempro_data should be TEMProTripEnds type not {type(tempro_data)}"
        )
    zone_translation_weights = {
        **dict.fromkeys(("hb_attractions", "nhb_attractions"), "employment"),
        **dict.fromkeys(("hb_productions", "nhb_productions"), "population"),
    }
    grown = {}
    for segment in dataclasses.fields(TEMProTripEnds):
        LOG.info("Calculating TEMPro trip end growth for %s", segment.name)
        grown[segment.name] = trip_end_growth(
            getattr(tempro_data, segment.name),
            model_zone_system,
            zone_translation_weights[segment.name],
        )
    return TEMProTripEnds(**grown)


def _trip_end_totals(
    name: str,
    row_targets: pd.Series,
    col_targets: pd.Series,
    tolerance: float = 1e-7,
) -> Dict[str, pd.Series]:
    """Compare `row_targets` and `col_targets` sum totals and factor if needed.

    If the totals for either of the targets differ by more than
    `tolerance` then both targets are factored to equal a mean
    total.

    Parameters
    ----------
    name : str
        Name of the trip end totals being factored,
        used in log messages.
    row_targets : pd.Series
        Target trip ends for the rows.
    col_targets : pd.Series
        Target trip ends for the columns.
    tolerance: float, default 1e-7
        Tolerance allowed when comparing the trip end totals.

    Returns
    -------
    Dict[str, pd.Series]
        Dictionary containing both targets with
        keys: 'row_targets' and 'col_targets'.
    """
    targets = {"row_targets": row_targets, "col_targets": col_targets}
    totals = {nm: s.sum() for nm, s in targets.items()}
    diff = abs(totals["row_targets"] - totals["col_targets"])
    if diff > tolerance:
        avg_tot = np.mean(list(totals.values()))
        LOG.debug(
            "%s row and column trip end totals differ by %.5e, "
            "factoring trip end totals to mean: %.5e",
            name,
            diff,
            avg_tot,
        )
        for nm, data in targets.items():
            targets[nm] = data * (avg_tot / totals[nm])
    return targets


def _check_matrix(
    matrix: pd.DataFrame,
    name: str,
    raise_nan_errors: bool = True,
):
    """Check if `matrix` contains any non-finite values and log some matrix statistics.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix to check.
    name : str
        Name of the matrix being checked (used in error message).
    raise_nan_errors : bool
        If check should raise errors (True) or just log them.

    Raises
    ------
    NTEMForecastError
        If any non-finite values are found and `raise_nan_errors`
        is True.
    """
    LOG.debug(
        "%s matrix: shape %s, total %.1e, mean %.1e",
        name,
        matrix.shape,
        np.sum(matrix.values),
        np.mean(matrix.values),
    )
    nans = np.sum(matrix.isna().values)
    infs = np.sum(np.isinf(matrix.values))
    if nans > 0 or infs > 0:
        err = (
            f"{name} matrix contains {nans:,} "
            f"NaN and {infs:,} infinite values"
        )
        if raise_nan_errors:
            raise NTEMForecastError(err)
        LOG.error(err)


def grow_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
    segment_name: str,
    attractions: nd_core.DVector,
    productions: nd_core.DVector,
) -> pd.DataFrame:
    """Grow `matrix` based on given growth factors and save to `output_path`.

    Internal growth is done using a 2D furness and
    external growth is done by factoring the rows,
    then the columns, to the targets.

    Parameters
    ----------
    matrix : pd.DataFrame
        Trip matrix for base year, columns and index
        should be zone numbers.
    output_path : Path
        Path to save the output file to.
    segment_name : str
        Name of the segment being grown, format
        `{purpose}_{mode}` e.g. `2_3`.
    attractions : nd_core.DVector
        DVector for the attractions growth factors,
        should be the same zone system as matrix.
    productions : nd_core.DVector
        DVector for the productions growth factors,
        should be the same zone system as matrix.

    Returns
    -------
    pd.DataFrame
        Trip `matrix` grown to match target `attractions`
        and `productions`.
    """
    # Calculate internal-internal target trip ends by
    # applying growth to matrix trip ends
    internals = attractions.zoning_system.internal_zones
    int_targets = {}
    growth = {}
    dvectors = ("row_targets", productions), ("col_targets", attractions)
    for nm, dvec in dvectors:
        mat_te = matrix.loc[internals, internals].sum(
            axis=1 if nm == "row_targets" else 0
        )
        mat_te.name = "base_trips"
        mat_te.index.name = "model_zone_id"
        growth[nm] = pd.Series(
            dvec.get_segment_data(segment_name),
            index=dvec.zoning_system.unique_zones,
            name="growth",
        )
        growth[nm].index.name = "model_zone_id"
        int_targets[nm] = pd.concat([growth[nm], mat_te], axis=1)
        int_targets[nm].loc[:, "trips"] = (
            int_targets[nm]["growth"] * int_targets[nm]["base_trips"]
        )
        int_targets[nm] = int_targets[nm].loc[internals, "trips"]

    # Distribute internal demand with 2D furnessing, targets
    # converted to DataFrames for this function
    int_targets = _trip_end_totals(segment_name, **int_targets)
    int_future, iters, rms = furness.furness_pandas_wrapper(
        matrix.loc[internals, internals],
        **{nm: data.reset_index() for nm, data in int_targets.items()},
        tol=1e-4,
        max_iters=3000,
    )
    LOG.debug(
        "Furnessed internal trips with %s iterations and RMS = %.1e",
        iters,
        rms,
    )
    # Factor external demand to row and column targets, make sure
    # row and column targets have the same totals
    ext_future = matrix.mul(growth["col_targets"], axis="columns")
    ext_future = ext_future.mul(growth["row_targets"], axis="index")
    # Set internal zones in factored matrix to 0 and add internal furnessed
    ext_future.loc[internals, internals] = 0
    combined_future = pd.concat([int_future, ext_future], axis=0)
    combined_future = combined_future.groupby(level=0).sum()
    _check_matrix(combined_future, output_path.stem)
    # Write future to file
    file_ops.write_df(combined_future, output_path)
    LOG.info("Written: %s", output_path)
    _pa_growth_comparison(
        {
            "base": matrix,
            "forecast": combined_future
        },
        {
            "attractions": growth["col_targets"],
            "productions": growth["row_targets"]
        },
        internals,
        output_path.with_name(output_path.stem + "-growth_comparison.xlsx"),
    )
    return combined_future


def _pa_growth_comparison(
    matrices: Dict[str, pd.DataFrame],
    growth_data: Dict[str, pd.Series],
    internals: np.ndarray,
    output_path: Path,
) -> Dict[str, Union[pd.DataFrame, float]]:
    """Write summary spreadsheet for PA matrix growth.

    Parameters
    ----------
    matrices : Dict[str, pd.DataFrame]
        Base and forecast PA matrices, should contain
        keys "base" and "forecast".
    growth_data : Dict[str, pd.Series]
        TEMPro trip end growth factors, should contain
        keys "attractions" and "productions".
    internals : np.ndarray
        Array containing the internal zone numbers.
    output_path : Path
        Path to Excel file to create.

    Returns
    -------
    Dict[str, Union[pd.DataFrame, float]]
        Dictionary containing various summary
        growth statistics that are written to
        the Excel file.

    Raises
    ------
    NTEMForecastError
        - If any of the input dictionaries don't
          contain the correct keys.
        - If the indices for the input matrices and
          growth factors aren't identical.
    """
    # Check dictionary keys and DataFrame/Series indices
    missing = [k for k in ("base", "forecast") if k not in matrices]
    if missing:
        raise NTEMForecastError(
            f"matrices dictionary is missing key(s): {missing}"
        )
    missing = [
        k for k in ("attractions", "productions") if k not in growth_data
    ]
    if missing:
        raise NTEMForecastError(
            f"growth_data dictionary is missing key(s): {missing}"
        )
    for nm, mat in matrices.items():
        if (mat.index != mat.columns).any():
            raise NTEMForecastError(
                f"{nm} matrix index and columns are not identical"
            )
    if (matrices["base"].index != matrices["forecast"].index).any():
        raise NTEMForecastError(
            "base and forecast matrices don't have indentical indices"
        )
    for nm, g in growth_data.items():
        if (g.index != matrices["base"].index).any():
            raise NTEMForecastError(
                f"{nm} growth does not have the same index as the matrix"
            )

    # Calculate aggregated growths
    growth_comparisons = {
        "Matrix Total Growth":
            (
                np.sum(matrices["forecast"].values) /
                np.sum(matrices["base"].values)
            ),
        "Matrix Mean Growth":
            np.nanmean((matrices["forecast"] / matrices["base"]).values),
    }
    for nm, g in growth_data.items():
        growth_comparisons[f"TEMPro Mean Growth - {nm.title()}"] = np.mean(
            g.values
        )

    # Reindex with internals/externals
    new_index = np.where(matrices["base"].index.isin(internals), "I", "E")
    matrix_te = {}
    for nm, mat in matrices.items():
        mat = mat.copy()
        mat.index = new_index
        mat.columns = new_index
        matrices[nm] = mat.groupby(level=0).sum().groupby(level=0, axis=1).sum()
        matrix_te[nm] = pd.DataFrame(
            {
                "Attractions": matrices[nm].sum(axis=1),
                "Productions": matrices[nm].sum(axis=0),
            },
            index=matrices[nm].index,
        )
        matrix_te[nm].loc["Total", :] = matrix_te[nm].sum()
    for nm, g in growth_data.items():
        g = g.copy()
        g.index = new_index
        growth_data[nm] = g.groupby(level=0).mean()

    growth_comparisons["Matrix Trip End Growth"] = (
        matrix_te["forecast"] / matrix_te["base"]
    )
    growth_comparisons["Matrix IE Growth"] = (
        matrices["forecast"] / matrices["base"]
    )
    growth_comparisons["TEMPro Trip End Growth"] = pd.DataFrame(growth_data)

    out = output_path.with_suffix(".xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as excel:
        pandas_types = (pd.DataFrame, pd.Series)
        single_values = {
            k: v
            for k, v in growth_comparisons.items()
            if not isinstance(v, pandas_types)
        }
        single_values = pd.Series(single_values)
        single_values.to_excel(excel, sheet_name="Summary", header=False)
        for nm, data in growth_comparisons.items():
            if not isinstance(data, pandas_types):
                continue
            data.to_excel(excel, sheet_name=nm)
    LOG.info("Written: %s", out)
    return growth_comparisons


def grow_all_matrices(
    matrices: NTEMImportMatrices,
    growth: TEMProTripEnds,
    output_folder: Path,
) -> None:
    """Grow all base year `matrices` to all forecast years in `trip_ends`.

    Parameters
    ----------
    matrices : NTEMImportMatrices
        Paths to the base year PostME matrices.
    trip_ends : TEMProTripEnds
        TEMPro growth factors for all forecast years,
        these factors should be in the same zone system
        as the `matrices`.
    model : str
        Name of the model e.g. 'noham'.
    output_folder : Path
        Path to folder for saving the output matrices.

    Raises
    ------
    NTEMForecastError
        If the productions and attractions in `trip_ends`
        don't have the same forecast years.

    See Also
    --------
    grow_matrix: for growing a single matrix.
    """
    iterator = {
        "hb": (
            matrices.hb_paths,
            growth.hb_attractions,
            growth.hb_productions,
        ),
        "nhb":
            (
                matrices.nhb_paths,
                growth.nhb_attractions,
                growth.nhb_productions,
            ),
    }
    output_folder.mkdir(exist_ok=True, parents=True)
    for hb, (paths, attractions, productions) in iterator.items():
        for purp, path in paths.items():
            LOG.info("Reading base year matrix: %s", path)
            base = file_ops.read_df(path, find_similar=True, index_col=0)
            base.columns = pd.to_numeric(base.columns, downcast="integer")
            for yr, attr in attractions.items():
                LOG.info("Growing %s to %s", path.stem, yr)
                try:
                    prod = productions[yr]
                except KeyError as err:
                    raise NTEMForecastError(
                        f"production trip ends doesn't contain year {yr}"
                    ) from err
                grow_matrix(
                    base,
                    output_folder / matrices.output_filename(hb, purp, yr),
                    f"{purp}_{matrices.mode}",
                    attr,
                    prod,
                )


def convert_to_od(
    pa_folder: Path,
    od_folder: Path,
    years: List[int],
    modes: List[int],
    purposes: Dict[str, List[int]],
    model_name: str,
    pa_to_od_factors: Dict[str, Path],
) -> None:
    """Converts PA matrices from folder to OD.

    Parameters
    ----------
    pa_folder : Path
        Path to folder containing PA matrices.
    od_folder : Path
        Path to folder to save OD matrices in.
    years : List[int]
        List of years of matrices to convert.
    modes : List[int]
        List of modes for matrices to convert.
    purposes : Dict[str, List[int]]
        Purposes to convert for home-based and non-home-based
        matrices, should have keys "hb" and "nhb".
    model_name : str
        Name of the model being ran.
    pa_to_od_factors : Dict[str, Path]
        Paths to the folders containing the PA to OD tour
        proportions, should have keys "post_me_fh_th_factors"
        and "post_me_tours".

    See Also
    --------
    pa_to_od.build_od_from_fh_th_factors : for home-based conversion
    matrix_processing.nhb_tp_split_via_factors : for non-home-based conversion
    """
    LOG.info("Converting PA to OD")
    od_folder.mkdir(exist_ok=True, parents=True)
    pa_to_od.build_od_from_fh_th_factors(
        pa_import=pa_folder,
        od_export=od_folder,
        fh_th_factors_dir=pa_to_od_factors["post_me_fh_th_factors"],
        years_needed=years,
        seg_level="tms",
        seg_params={
            "p_needed": purposes["hb"],
            "m_needed": modes
        },
    )
    matrix_processing.nhb_tp_split_via_factors(
        import_dir=pa_folder,
        export_dir=od_folder,
        import_matrix_format="pa",
        export_matrix_format="od",
        tour_proportions_dir=pa_to_od_factors["post_me_tours"],
        model_name=model_name,
        years_needed=years,
        p_needed=purposes["nhb"],
        m_needed=modes,
        compress_out=True,
    )


def compile_noham_for_norms(pa_folder: Path, years: List[int]) -> Path:
    """Compile the PA matrices into the 24hr VDM PA matrices format.

    The outputs are saved in a new folder called
    "24hr PA Matrices" with the same parent as `pa_folder`.

    Parameters
    ----------
    pa_folder : Path
        Folder containing PA matrices.
    years : List[int]
        List of the years to convert.

    Returns
    -------
    vdm_folder : Path
        Folder where the output 24hr VDM PA Matrices are saved.
    """
    vdm_folder = pa_folder.with_name("24hr VDM PA Matrices")
    vdm_folder.mkdir(exist_ok=True)

    paths = matrix_processing.build_compile_params(
        import_dir=pa_folder,
        export_dir=vdm_folder,
        matrix_format="pa",
        years_needed=years,
        m_needed=[3],
        split_hb_nhb=True,
    )
    for path in paths:
        matrix_processing.compile_matrices(
            mat_import=pa_folder,
            mat_export=vdm_folder,
            compile_params_path=path,
        )
    LOG.info("Written 24hr VDM PA Matrices: %s", vdm_folder)
    return vdm_folder


def compile_noham(
    import_od_path: Path, years: List[int], car_occupancies_path: Path
) -> Path:
    """Compile OD matrices into the formats required for NoHAM.

    Parameters
    ----------
    import_od_path : Path
        Folder containing the existing OD matrices.
    years : List[int]
        List of years to compile.
    car_occupancies_path : Path
        Path to CSV containing car occupancies.

    Returns
    -------
    Path
        Path to the Compiled OD matrices folder, will
        be named "Compiled OD" and in the same folder
        as `import_od_path`.
    """
    compiled_od_path = import_od_path.with_name("Compiled OD")
    compiled_od_pcu_path = compiled_od_path / "PCU"
    compiled_od_pcu_path.mkdir(parents=True, exist_ok=True)

    compile_params_paths = matrix_processing.build_compile_params(
        import_dir=import_od_path,
        export_dir=compiled_od_path,
        matrix_format="od",
        years_needed=years,
        m_needed=[3],
        tp_needed=[1, 2, 3, 4],
    )
    for path in compile_params_paths:
        matrix_processing.compile_matrices(
            mat_import=import_od_path,
            mat_export=compiled_od_path,
            compile_params_path=path,
        )
    LOG.info("Written Compiled OD matrices: %s", compiled_od_path)
    car_occupancies = pd.read_csv(car_occupancies_path)

    # Need to convert into hourly average PCU for noham
    vehicle_occupancy.people_vehicle_conversion(
        mat_import=compiled_od_path,
        mat_export=compiled_od_pcu_path,
        car_occupancies=car_occupancies,
        mode=3,
        method="to_vehicles",
        out_format="wide",
        hourly_average=True,
    )
    LOG.info("Written Compiled OD PCU matrices: %s", compiled_od_pcu_path)
    return compiled_od_path


def _filename_contents(filename: str) -> Dict[str, Any]:
    """Extract information from matrix filenames.

    Parameters
    ----------
    filename : str
        Filename to extract information form.

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
        r"(?P<matrix_type>nhb|hb)"
        r"_(?P<trip_end_type>pa|od)"
        r"_yr(?P<year>\d{4})"
        r"_p(?P<purpose>\d{,2})"
        r"_m(?P<mode>\d)",
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
    for p in pa_folder.iterdir():
        if p.is_dir():
            continue
        try:
            file_data = _filename_contents(p.name)
        except NTEMForecastError as err:
            LOG.error(err)
        file_data["path"] = p
        files.append(file_data)
    files = pd.DataFrame(files)

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
