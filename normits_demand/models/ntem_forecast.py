# -*- coding: utf-8 -*-
"""
    Module for producing forecast demand constrained to NTEM.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
from pathlib import Path
from typing import Dict, Any, List

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from normits_demand.utils import file_ops
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand import efs_constants as efs_consts
from normits_demand.distribution import furness
from normits_demand.models.tempro_trip_ends import NTEMForecastError, TEMProTripEnds
from normits_demand.matrices import pa_to_od, matrix_processing

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)
LAD_ZONE_SYSTEM = "lad_2020"


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
    tempro_vectors: Dict[int, nd_core.DVector]
) -> Dict[int, nd_core.DVector]:
    """Calculate growth at LAD level and return it a `tempro_vectors` zone system.

    The trip ends are translated to `LAD_ZONE_SYSTEM` to
    calculate growth factors then translated back to the
    original zone system before returning.

    Parameters
    ----------
    tempro_vectors : Dict[int, nd_core.DVector]
        Trip end vectors from TEMPro for all study years,
        keys should be years and must include
        `normits_demand.efs_constants.BASE_YEAR`.

    Returns
    -------
    Dict[int, nd_core.DVector]
        Trip end growth factors in same zone system as
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
    old_zone = tempro_vectors[efs_consts.BASE_YEAR].zoning_system
    growth_zone = nd_core.get_zoning_system(LAD_ZONE_SYSTEM)
    base_data = tempro_vectors[efs_consts.BASE_YEAR
                              ].translate_zoning(growth_zone)
    # Convert to LADs and calculate growth from base year
    growth = {}
    # Ignore divide by zero warnings and fill with zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        for yr, data in tempro_vectors.items():
            if yr == efs_consts.BASE_YEAR:
                continue
            data = data.translate_zoning(growth_zone) / base_data
            # Set any nan or inf values created by dividing by 0 to 0 growth
            data = data.segment_apply(
                np.nan_to_num, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Translate back to original zone system, without using
            # weighting factors i.e. weighting factors of 1
            growth[yr] = data.translate_zoning(old_zone, weighting="no_weight")
    return growth


def grow_trip_ends(
    tempro_vectors: Dict[int, nd_core.DVector]
) -> Dict[int, nd_core.DVector]:
    """Grow TEMPro trip ends based on LAD growth.

    Growth factors are calculated at `LAD_ZONE_SYSTEM`
    level but applied at the original zone system.

    Parameters
    ----------
    tempro_vectors : Dict[int, nd_core.DVector]
        Trip end vectors from TEMPro for all study years,
        keys should be years and must include
        `normits_demand.efs_constants.BASE_YEAR`.

    Returns
    -------
    Dict[int, nd_core.DVector]
        Future trip ends in same zone system as `tempro_vectors`
        base year, contains all years from `tempro_vectors`
        except the base year.

    Raises
    ------
    ValueError
        If `normits_demand.efs_constants.BASE_YEAR` is not
        in `tempro_vectors`.

    See Also
    --------
    trip_end_growth : for growth factor calculation
    """
    # Calculate growth at LAD level
    te_growth = trip_end_growth(tempro_vectors)
    base_data = tempro_vectors[efs_consts.BASE_YEAR]
    future = {}
    for yr, growth in te_growth.items():
        future[yr] = base_data * growth
    return future


def grow_tempro_data(tempro_data: TEMProTripEnds) -> TEMProTripEnds:
    """Calculate LAD growth factors and use them to grow `tempro_data`.

    Growth factors are calculated at LAD level but are applied
    to the `tempro_data` at MSOA level.

    Parameters
    ----------
    tempro_data : TEMProTripEnds
        TEMPro trip end data for all study years.

    Returns
    -------
    TEMProTripEnds
        Forecasted TEMPro trip ends for all future
        years.

    Raises
    ------
    NTEMForecastError
        If `tempro_data` isn't an instance of `TEMProTripEnds`.

    See Also
    --------
    grow_trip_ends: which will grow trip ends for all years for
        a single dictionary e.g. `hb_attractions`.
    """
    if not isinstance(tempro_data, TEMProTripEnds):
        raise NTEMForecastError(
            f"tempro_data should be TEMProTripEnds type not {type(tempro_data)}"
        )
    grown = {}
    for segment in dataclasses.fields(TEMProTripEnds):
        LOG.info("Calculating TEMPro trip end growth for %s", segment.name)
        grown[segment.name] = grow_trip_ends(getattr(tempro_data, segment.name))
    return TEMProTripEnds(**grown)


def _trip_end_totals(
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
            "Row and column trip end totals differ by %.5e, "
            "factoring trip end totals to mean: %.5e",
            diff,
            avg_tot,
        )
        for nm, data in targets.items():
            targets[nm] = data * (avg_tot / totals[nm])
    return targets


def grow_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
    segment_name: str,
    attractions: nd_core.DVector,
    productions: nd_core.DVector,
) -> pd.DataFrame:
    """Grow `matrix` to trip end totals and save to `output_path`.

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
        DVector for the attractions trip ends,
        should be the same zone system as matrix.
    productions : nd_core.DVector
        DVector for the productions trip ends,
        should be the same zone system as matrix.

    Returns
    -------
    pd.DataFrame
        Trip `matrix` grown to match target `attractions`
        and `productions`.
    """
    # Get single segment as a Series from DVectors
    targets = {}
    dvectors = {"row_targets": productions, "col_targets": attractions}
    for nm, dvec in dvectors.items():
        targets[nm] = pd.Series(
            dvec.get_segment_data(segment_name),
            index=dvec.zoning_system.unique_zones,
            name="trips",
        )
        targets[nm].index.name = "model_zone_id"

    # Get internal targets only and factor to the same totals
    internals = attractions.zoning_system.internal_zones
    int_targets = {nm: data.loc[internals] for nm, data in targets.items()}
    int_targets = _trip_end_totals(**int_targets)
    # Distribute internal demand with 2D furnessing, targets
    # converted to DataFrames for this function
    int_future, iters, r_sq = furness.furness_pandas_wrapper(
        matrix.loc[internals, internals],
        **{nm: data.reset_index() for nm, data in int_targets.items()},
    )
    LOG.debug(
        "Furnessed internal trips with %s iterations and R^2 = %.2f",
        iters,
        r_sq,
    )
    # Factor external demand to row and column targets, make sure
    # row and column targets have the same totals
    targets = _trip_end_totals(**targets)
    row_factors = targets["row_targets"] / matrix.sum(axis=0)
    ext_future = matrix.mul(row_factors.fillna(0), axis="index")
    col_factors = targets["col_targets"] / ext_future.sum(axis=1)
    ext_future = ext_future.mul(col_factors.fillna(0), axis="columns")
    # Set internal zones in factored matrix to 0 and add internal furnessed
    ext_future.loc[internals, internals] = 0
    combined_future = ext_future + int_future
    # Write future to file
    file_ops.write_df(combined_future, output_path)
    LOG.info("Written: %s", output_path)
    return combined_future


def grow_all_matrices(
    matrices: NTEMImportMatrices,
    trip_ends: TEMProTripEnds,
    model: str,
    output_folder: Path,
) -> None:
    """Grow all base year `matrices` to all forecast years in `trip_ends`.

    Parameters
    ----------
    matrices : NTEMImportMatrices
        Paths to the base year PostME matrices.
    trip_ends : TEMProTripEnds
        TEMPro trip end data for all forecast years.
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
    zone_translation_weights = {
        **dict.fromkeys(("hb_attractions", "nhb_attractions"), "employment"),
        **dict.fromkeys(("hb_productions", "nhb_productions"), "population"),
    }
    trip_ends = trip_ends.translate_zoning(
        model, weighting=zone_translation_weights
    )
    iterator = {
        "hb":
            (
                matrices.hb_paths,
                trip_ends.hb_attractions,
                trip_ends.hb_productions,
            ),
        "nhb":
            (
                matrices.nhb_paths,
                trip_ends.nhb_attractions,
                trip_ends.nhb_productions,
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
    )
