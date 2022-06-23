# -*- coding: utf-8 -*-
"""Script for producing maps and graphs for the NTEM forecasting report."""

##### IMPORTS #####
from __future__ import annotations

# Standard imports
import collections
import dataclasses
import enum
import functools as ft
import itertools
import math
import os
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

# Third party imports
import geopandas as gpd
import mapclassify
import matplotlib.backends.backend_pdf as backend_pdf
import numpy as np
import openpyxl
import pandas as pd
from matplotlib import cm, colors, figure, lines, patches
from matplotlib import pyplot as plt
from matplotlib import ticker
from openpyxl.worksheet.datavalidation import DataValidation, DataValidationList
from scipy import stats
from shapely import geometry

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import colours as tfn_colours
from normits_demand import logging as nd_log
from normits_demand.core import enumerations as nd_enum
from normits_demand.reports import ntem_forecast_checks
from normits_demand.utils import file_ops, pandas_utils

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".ntem_plots")
TRIP_ORIGINS = {"hb": "Home-based", "nhb": "Non-home-based"}
TFN_COLOURS = [
    tfn_colours.TFN_NAVY,
    tfn_colours.TFN_TEAL,
    tfn_colours.TFN_PURPLE,
    tfn_colours.TFN_PINK,
]

##### CLASSES #####
@dataclasses.dataclass
class MatrixTripEnds:
    """Dataclass for storing matrix productions and attractions trip ends."""

    productions: pd.DataFrame
    attractions: pd.DataFrame

    def _check_other(self, other: Any, operation: str) -> None:
        if not isinstance(other, type(self)):
            raise TypeError(f"cannot perform {operation} with {type(other)}")

    def __add__(self, other: MatrixTripEnds) -> MatrixTripEnds:
        """Add productions and attractions from `other` to self."""
        self._check_other(other, "addition")
        return MatrixTripEnds(
            self.productions + other.productions, self.attractions + other.attractions
        )

    def __sub__(self, other: MatrixTripEnds) -> MatrixTripEnds:
        """Subtract productions and attractions from `other` from self."""
        self._check_other(other, "subtraction")
        return MatrixTripEnds(
            self.productions - other.productions, self.attractions - other.attractions
        )

    def __truediv__(self, other: MatrixTripEnds) -> MatrixTripEnds:
        """Divide productions and attractions from self by `other`."""
        self._check_other(other, "division")
        return MatrixTripEnds(
            self.productions / other.productions, self.attractions / other.attractions
        )


class GeoSpatialFile(NamedTuple):
    """Path to a geospatial file and the relevant ID column name."""

    path: Path
    id_column: str


@dataclasses.dataclass
class PAPlotsParameters:
    """Parameters for producing the NTEM comparison plots."""

    base_matrix_folder: Path
    forecast_matrix_folder: Path
    matrix_zoning: str
    plot_zoning: str
    output_folder: Path
    geospatial_file: GeoSpatialFile
    analytical_area_shape: GeoSpatialFile
    tempro_comparison_folder: Path
    tempro_comparison_summary_zoning: str
    base_year: int


class PlotType(nd_enum.AutoName):
    """Plot type options for use in `matrix_total_plots`."""

    BAR = enum.auto()
    LINE = enum.auto()


@dataclasses.dataclass
class CustomCmap:
    """Store information about a custom colour map."""

    bin_categories: pd.Series
    colours: pd.DataFrame
    legend_elements: list[patches.Patch]

    def __add__(self, other: CustomCmap) -> CustomCmap:
        """Return new CustomCmap with the attributes from `self` and `other` concatenated."""
        if not isinstance(other, CustomCmap):
            raise TypeError(f"other should be a CustomCmap not {type(other)}")
        return CustomCmap(
            pd.concat([self.bin_categories, other.bin_categories], verify_integrity=True),
            pd.concat([self.colours, other.colours], verify_integrity=True),
            self.legend_elements + other.legend_elements,
        )


##### FUNCTIONS #####
def match_files(folder: Path, pattern: re.Pattern) -> Iterator[tuple[dict[str, str], Path]]:
    """Iterate through all files in folder which match `pattern`.

    Parameters
    ----------
    folder : Path
        Folder to find files in.
    pattern : re.Pattern
        Pattern that the filename should match.

    Yields
    ------
    dict[str, str]
        Dictionary of `pattern` groups and their values.
    Path
        Path to the file.
    """
    for file in folder.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.stem)
        if match is None:
            continue
        yield match.groupdict(), file


def get_matrix_totals(
    matrix: pd.DataFrame, zoning_name: str, trip_origin: str, user_class: str, year: str
) -> pd.DataFrame:
    """Calculate the matrix II, IE, EI, EE trip end totals.

    Parameters
    ----------
    matrix : pd.DataFrame
        PA matrix.
    zoning_name : str
        Name of matrix zone system
    trip_origin : str
        Name of trip origin type 'hb' or 'nhb'.
    user_class : str
        Name of the user class.
    year : str
        Matrix year.

    Returns
    -------
    pd.DataFrame
        Trip end matrix totals by matrix area type.
    """
    zoning = nd.get_zoning_system(zoning_name)
    totals = pandas_utils.internal_external_report(
        matrix, zoning.internal_zones, zoning.external_zones
    )
    totals = totals.unstack()
    totals = totals.drop(
        index=[
            ("internal", "total"),
            ("external", "total"),
            ("total", "internal"),
            ("total", "external"),
        ]
    )
    totals.index = totals.index.get_level_values(0) + "-" + totals.index.get_level_values(1)
    totals.index = totals.index.str.replace("total-total", "total")

    # Check sum adds up correctly
    index_check = [
        "internal-internal",
        "internal-external",
        "external-internal",
        "external-external",
    ]
    abs_diff = np.abs(totals.at["total"] - totals.loc[index_check].sum())
    if abs_diff > 1e-5:
        LOG.warning(
            "matrix II, IE, EI and EE don't sum to matrix total, absolue difference is %.1e",
            abs_diff,
        )

    return pd.DataFrame(
        {
            "trip_origin": trip_origin,
            "user_class": user_class,
            "matrix_area": totals.index,
            year: totals.values,
        }
    )


def matrix_trip_ends(
    matrix: Path, matrix_zoning: str, to_zoning: str = None, **kwargs
) -> tuple[MatrixTripEnds, pd.DataFrame]:
    """Calculate the matrix trip ends and totals for a given matrix file.

    Matrix trip ends are returned in the `to_zoning` system.

    Parameters
    ----------
    matrix : Path
        Path to PA matrix CSV.
    matrix_zoning : str
        Zoning system of the `matrix`.
    to_zoning : str, optional
        Zoning system to return the trip ends in.

    Returns
    -------
    MatrixTripEnds
        Productions and attractions trip ends at `to_zoning`.
    pd.DataFrame
        Productions and attractions totals.
    """
    mat = file_ops.read_df(matrix, find_similar=True, index_col=0)
    mat.columns = pd.to_numeric(mat.columns, downcast="integer")
    totals = get_matrix_totals(mat, matrix_zoning, **kwargs)
    if to_zoning:
        mat = ntem_forecast_checks.translate_matrix(mat, matrix_zoning, to_zoning)
        matrix_zoning = to_zoning
    return MatrixTripEnds(mat.sum(axis=1), mat.sum(axis=0)), totals


def get_base_trip_ends(
    folder: Path, matrix_zoning: str, plot_zoning: str
) -> tuple[dict[tuple[str, str], MatrixTripEnds], pd.DataFrame]:
    """Read matrices from `folder` and return trip ends.

    Parameters
    ----------
    folder : Path
        Folder containing NTEM PA matrices.
    matrix_zoning : str
        Zoning system of the matrix files.
    plot_zoning : str
        Zoning system to convert the trip ends to.

    Returns
    -------
    dict[tuple[str, str], MatrixTripEnds]
        Dictionary containing all the matrix trip ends at `plot_zoning`
        with trip origin and user class as the keys.
    pd.DataFrame
        Dataframe containing the trip end totals for all matrices.
    """
    UC_LOOKUP = collections.defaultdict(
        lambda: "other", {**dict.fromkeys((2, 12), "business"), 1: "commute"}
    )
    LOG.info("Extracting base trip ends from %s", folder)
    file_pattern = re.compile(
        r"(?P<trip_origin>n?hb)_pa"
        r"_yr(?P<year>\d{4})"
        r"_p(?P<purpose>\d{1,2})"
        r"_m(?P<mode>\d{1,2})",
        re.IGNORECASE,
    )
    matrix_totals = []
    trip_ends = collections.defaultdict(list, {})
    for params, file in match_files(folder, file_pattern):
        uc = UC_LOOKUP[int(params.pop("purpose"))]
        te, total = matrix_trip_ends(
            file,
            matrix_zoning,
            plot_zoning,
            trip_origin=params["trip_origin"],
            user_class=uc,
            year=params["year"],
        )
        matrix_totals.append(total)
        trip_ends[(params["trip_origin"], uc)].append(te)

    for key, te in trip_ends.items():
        trip_ends[key] = sum(te, start=MatrixTripEnds(0, 0))

    return trip_ends, pd.concat(matrix_totals)


def _plot_bars(
    ax: plt.Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    colour: str,
    width: float,
    label: str,
    max_height: float,
    label_fmt: str = ".3g",
):
    """Create the bar plots used in `matrix_total_plots`."""
    bars = ax.bar(x_data, y_data, label=label, color=colour, width=width, align="edge",)

    for rect in bars:
        height = rect.get_height()
        if height > (0.2 * max_height):
            y_pos = height * 0.8
        else:
            y_pos = height + (0.1 * max_height)
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y_pos,
            f"{height:{label_fmt}}",
            ha="center",
            va="center",
            rotation=45,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )


def _plot_line(
    ax: plt.Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    colour: str,
    label: str,
    label_fmt: str = ".3g",
):
    """Create the line plots used in `matrix_total_plots`."""
    ax.plot(x_data, y_data, label=label, color=colour, marker="+")

    for x, y in zip(x_data, y_data):
        if x < x_data[-1]:
            text_offset = (10, 0)
            ha = "left"
        else:
            text_offset = (-10, 0)
            ha = "right"
        ax.annotate(
            f"{y:{label_fmt}}",
            (x, y),
            xytext=text_offset,
            textcoords="offset pixels",
            ha=ha,
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )


def matrix_total_plots(
    totals: pd.DataFrame, title: str, ylabel: str, plot_type: PlotType = PlotType.BAR
) -> figure.Figure:
    """Create single graph for trip end totals and return figure.

    Parameters
    ----------
    totals : pd.DataFrame
        DataFrame containing trip end totals for HB/NHB productions and attractions.
    title : str
        Title for the graphs.
    ylabel : str
        Label on the Y axis.
    plot_type : PlotType, default PlotType.BAR
        Whether to do a bar or line plot.

    Returns
    -------
    figure.Figure
        Matplotlib figure with 2 axes.

    Raises
    ------
    ValueError
        If `plot_type` isn't valid.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 15), tight_layout=True)
    fig.suptitle(title)
    colours = dict(zip(("business", "commute", "other"), TFN_COLOURS))

    FULL_WIDTH = 0.9
    ax: plt.Axes
    for to, ax in zip(TRIP_ORIGINS.keys(), axes):
        rows = totals.loc[to]
        x = np.arange(len(totals.columns))
        width = FULL_WIDTH / len(rows)
        max_height = np.max(rows.values)
        fmt = ".3g" if max_height < 1000 else ".2g"
        for i, (uc, row) in enumerate(rows.iterrows()):
            kwargs = dict(
                colour=colours[uc], label=f"{to.upper()} - {uc.title()}", label_fmt=fmt,
            )
            if plot_type == PlotType.BAR:
                _plot_bars(
                    ax, x + i * width, row.values, width=width, max_height=max_height, **kwargs
                )
            elif plot_type == PlotType.LINE:
                years = pd.to_numeric(row.index, downcast="integer")
                _plot_line(ax, years, row.values, **kwargs)
            else:
                raise ValueError(f"plot_type should be PlotType not {plot_type}")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        ax.set_title(TRIP_ORIGINS[to].title())
        if plot_type == PlotType.BAR:
            ax.set_xticks(x + (FULL_WIDTH / 2))
            ax.set_xticklabels(totals.columns.str.replace("2018", "Base (2018)"))
        elif plot_type == PlotType.LINE:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()

    return fig


def plot_all_matrix_totals(
    totals: pd.DataFrame, out_path: Path,
):
    """Plot the matrix trip end totals and growths.

    Parameters
    ----------
    totals : pd.DataFrame
        DataFrame containing the matrix trip end totals at various
        spatial aggregations.
    out_path : Path
        Path to save the PDF containing all the plots.
    """
    out_path = out_path.with_suffix(".pdf")
    with backend_pdf.PdfPages(out_path) as pdf:
        for area in totals.index.get_level_values(0).unique():
            curr_tot = totals.loc[area]
            total_figure = matrix_total_plots(
                curr_tot,
                f"NTEM Forecasting PA - {area.title()} Matrix Totals",
                "Matrix Trips",
            )
            pdf.savefig(total_figure)

            growth_figure = matrix_total_plots(
                curr_tot.div(curr_tot["2018"], axis=0),
                f"NTEM Forecasting PA - {area.title()} Matrix Growth",
                "Matrix Growth",
                plot_type=PlotType.LINE,
            )
            pdf.savefig(growth_figure)

    LOG.info("Written: %s", out_path)


def get_geo_data(file: GeoSpatialFile) -> gpd.GeoSeries:
    """Read shapefile data, set index to given ID column and convert CRS to EPSG=27700."""
    geo = gpd.read_file(file.path)
    if file.id_column not in geo.columns:
        raise KeyError(f"{file.id_column} missing from {file.path.name}")
    geo = geo.set_index(file.id_column)
    geo = geo.to_crs(27700)
    return geo["geometry"]


def _heatmap_figure(
    geodata: gpd.GeoDataFrame,
    column_name: str,
    title: str,
    bins: Optional[List[Union[int, float]]] = None,
    n_bins: int = 5,
    analytical_area: Union[geometry.Polygon, geometry.MultiPolygon] = None,
    positive_negative_colormaps: bool = False,
):
    LEGEND_KWARGS = dict(title_fontsize="large", fontsize="medium")

    fig, axes = plt.subplots(1, 2, figsize=(20, 15), frameon=False, constrained_layout=True)
    fig.suptitle(title, fontsize="xx-large")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_axis_off()
        if analytical_area is not None:
            add_analytical_area(ax, analytical_area)
    if analytical_area is not None:
        axes[0].legend(**LEGEND_KWARGS, loc="upper right")

    # Drop nan
    geodata = geodata.loc[~geodata[column_name].isna()]

    kwargs = dict(
        column=column_name,
        cmap="viridis_r",
        scheme="NaturalBreaks",
        k=7,
        legend_kwds=dict(
            title=f"{str(column_name).title()}", **LEGEND_KWARGS, loc="upper right",
        ),
        # missing_kwds={
        #     "color": "lightgrey",
        #     "edgecolor": "red",
        #     "hatch": "///",
        #     "label": "Missing values",
        # },
        linewidth=0.0,
        edgecolor="black",
    )

    if positive_negative_colormaps:
        # Calculate, and apply, separate colormaps for positive and negative values
        label_fmt = "{:.1%}"
        negative_cmap = _colormap_classify(
            geodata.loc[geodata[column_name] <= 0, column_name],
            "PuBu_r",
            label_fmt=label_fmt,
            n_bins=n_bins,
            bins=list(filter(lambda x: x <= 0, bins)) if bins is not None else bins,
        )
        positive_cmap = _colormap_classify(
            geodata.loc[geodata[column_name] > 0, column_name],
            "YlGn",
            label_fmt=label_fmt,
            n_bins=n_bins,
            bins=list(filter(lambda x: x > 0, bins)) if bins is not None else bins,
        )
        cmap = negative_cmap + positive_cmap
        # Update colours index to be the same order as geodata
        cmap.colours = cmap.colours.reindex(geodata.index)

        for ax in axes:
            geodata.plot(
                ax=ax,
                color=cmap.colours.values,
                linewidth=kwargs["linewidth"],
                edgecolor=kwargs["edgecolor"],
            )
        axes[1].legend(handles=cmap.legend_elements, **kwargs.pop("legend_kwds"))

    else:
        if bins:
            kwargs["scheme"] = "UserDefined"
            bins = sorted(bins)
            max_ = np.max(geodata[column_name].values)
            if bins[-1] < max_:
                bins[-1] = math.ceil(max_)
            kwargs["classification_kwds"] = {"bins": bins}
            del kwargs["k"]

        # If the quatiles scheme throws a warning then use FisherJenksSampled
        warnings.simplefilter("error", category=UserWarning)
        try:
            geodata.plot(ax=axes[0], legend=False, **kwargs)
            geodata.plot(ax=axes[1], legend=True, **kwargs)
        except UserWarning:
            kwargs["scheme"] = "FisherJenksSampled"
            geodata.plot(ax=axes[0], legend=False, **kwargs)
            geodata.plot(ax=axes[1], legend=True, **kwargs)
        finally:
            warnings.simplefilter("default", category=UserWarning)

        # Format legend text
        legend = axes[1].get_legend()
        for label in legend.get_texts():
            text = label.get_text()
            values = [float(s.strip()) for s in text.split(",")]
            lower = np.floor(values[0] * 100) / 100
            upper = np.ceil(values[1] * 100) / 100
            # Set to 0 if 0 or -0
            lower = 0 if lower == 0 else lower
            upper = 0 if upper == 0 else upper

            if lower == -np.inf:
                text = f"< {upper:.0%}"
            elif upper == np.inf:
                text = f"> {lower:.0%}"
            else:
                text = f"{lower:.0%} - {upper:.0%}"
            label.set_text(text)

    axes[1].set_xlim(300000, 600000)
    axes[1].set_ylim(150000, 500000)
    axes[1].annotate(
        "Source: NorMITs Demand",
        xy=(0.9, 0.01),
        xycoords="figure fraction",
        bbox=dict(boxstyle="square", fc="white"),
    )
    return fig


def trip_end_growth_heatmap(
    geospatial: gpd.GeoSeries,
    base: MatrixTripEnds,
    forecast: MatrixTripEnds,
    output_path: Path,
    title: str,
    bins: list[Union[int, float]] = None,
    analytical_area: Union[geometry.Polygon, geometry.MultiPolygon] = None,
):
    """Create heatmap of the trip end growth.

    Parameters
    ----------
    geospatial : gpd.GeoSeries
        Polygons for creating the heatmap.
    base : MatrixTripEnds
        Trip ends for the base matrix.
    forecast : MatrixTripEnds
        Trip ends for the forecast matrix.
    output_path : Path
        Path to save the output to.
    title : str
        Title to use for the plots.
    bins : list[Union[int, float]], optional
        Bands to use for the heat map if not given will
        calculate appropriate bins.
    analytical_area : Union[geometry.Polygon, geometry.MultiPolygon], optional
        Polygon to add to the map showing the analytical area boundary.
    """
    growth = (forecast / base) - MatrixTripEnds(1, 1)

    with backend_pdf.PdfPages(output_path) as pdf:
        for field in dataclasses.fields(MatrixTripEnds):
            geodata = getattr(growth, field.name)
            name = f"{field.name.title()} Growth"
            geodata.name = name
            geodata = gpd.GeoDataFrame(
                pd.concat([geospatial, geodata], axis=1),
                crs=geospatial.crs,
                geometry="geometry",
            )

            fig = _heatmap_figure(
                geodata, name, title, bins, analytical_area, positive_negative_colormaps=True
            )
            pdf.savefig(fig)
            plt.close()

    LOG.info("Saved: %s", output_path)


def ntem_pa_plots(
    base_folder: Path,
    forecast_folder: Path,
    matrix_zoning: str,
    plot_zoning: str,
    out_folder: Path,
    geospatial_file: GeoSpatialFile,
    analytical_area_file: GeoSpatialFile,
):
    """Create PA trip end growth graphs and maps.

    Parameters
    ----------
    base_folder : Path
        Folder containing the base PA matrices.
    forecast_folder : Path
        Folder containing the forecast PA matrices.
    matrix_zoning : str
        Zoning system that the matrices are in.
    plot_zoning : str
        Zoning system to output the maps in.
    out_folder : Path
        Folder to save the output maps and graphs to.
    geospatial_file : GeoSpatialFile
        File containing the spatial data for creating the maps.
    analytical_area_file : GeoSpatialFile
        File containing the spatial data for the analytical area boundary.
    """
    warnings.filterwarnings(
        "ignore", message=".*zones in the matrix are missing", category=UserWarning
    )
    base_trip_ends, base_totals = get_base_trip_ends(base_folder, matrix_zoning, plot_zoning)
    index_cols = ["matrix_area", "trip_origin", "user_class"]
    base_totals = base_totals.groupby(index_cols).sum()

    geospatial = get_geo_data(geospatial_file)
    analytical_area = get_geo_data(analytical_area_file).iloc[0]

    # Iterate through files which have the correct name pattern
    LOG.info("Extracting forecast trip ends from %s", forecast_folder)
    uc_str = "|".join(("business", "commute", "other"))
    file_pattern = re.compile(
        r"(?P<trip_origin>n?hb)_pa_"
        rf"(?P<user_class>{uc_str})"
        r"_yr(?P<year>\d{4})"
        r"_m(?P<mode>\d{1,2})",
        re.IGNORECASE,
    )
    forecast_totals = []
    for params, file in match_files(forecast_folder, file_pattern):
        trip_ends, total = matrix_trip_ends(
            file,
            matrix_zoning,
            plot_zoning,
            trip_origin=params["trip_origin"],
            user_class=params["user_class"],
            year=params["year"],
        )
        forecast_totals.append(total)

        base_key = (params["trip_origin"], params["user_class"])
        trip_end_growth_heatmap(
            geospatial,
            base_trip_ends[base_key],
            trip_ends,
            out_folder
            / "NTEM_forecast_growth_{}_{}_{}-{}.pdf".format(
                params["year"], *base_key, plot_zoning
            ),
            "NTEM Forecast {} - {} {}".format(
                params["year"], params["trip_origin"].upper(), params["user_class"].title(),
            ),
            analytical_area=analytical_area,
            bins=[-1, -0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5, 1, np.inf],
        )

    forecast_totals: pd.DataFrame = pd.concat(forecast_totals)
    forecast_totals = forecast_totals.groupby(index_cols).agg(np.nansum)
    plot_all_matrix_totals(
        pd.concat([base_totals, forecast_totals], axis=1),
        out_folder / "NTEM_PA_matrix_totals.pdf",
    )


def _linear_fit(data: pd.DataFrame, ax: plt.Axes, color: str, label: str) -> lines.Line2D:
    """Plot line of best fit on `ax`."""
    fit = stats.linregress(data[["matrix", "tempro"]].values)
    fit_x = (np.min(data["matrix"]), np.max(data["matrix"]))
    fit_y = [x * fit.slope + fit.intercept for x in fit_x]

    line = ax.plot(
        fit_x,
        fit_y,
        "--",
        color=color,
        alpha=0.7,
        label=f"{label}\n$y = {fit.slope:.2f}x {fit.intercept:+.1f}$"
        f"\n$R^2={fit.rvalue**2:.2f}$",
    )[0]
    return line


def growth_comparison_regression(growth: pd.DataFrame, output_path: Path, title: str) -> None:
    """Create NTEM model vs TEMPro trip end growth comparison plot.

    Parameters
    ----------
    growth : pd.DataFrame
        DataFrame with NTEM and TEMPro trip end growth, expects
        columns: `zone`, `trip_origin`, `pa`, `matrix_growth`,
        `tempro_growth` and `matrix_forecast`.
    output_path : Path
        Path to save the output PDF to.
    title : str
        Title of the graphs.
    """
    expected_columns = [
        "zone",
        "trip_origin",
        "pa",
        "IE",
        "matrix_growth",
        "tempro_growth",
        "matrix_forecast",
    ]
    growth = growth.reset_index().set_index(expected_columns[:4]).loc[:, expected_columns[4:]]
    growth.rename(columns={"matrix_growth": "matrix", "tempro_growth": "tempro"}, inplace=True)
    growth.dropna(inplace=True)

    with backend_pdf.PdfPages(output_path) as pdf:

        for to in nd.TripOrigin:
            for pa in ("productions", "attractions"):
                fig, axd = plt.subplot_mosaic(
                    [["internal", "colorbar"], ["external", "colorbar"],],
                    gridspec_kw=dict(width_ratios=[1, 0.05]),
                    figsize=(9, 15),
                    tight_layout=True,
                )

                filtered = growth.loc[:, to.get_name(), pa, :]

                # Use different markers for internal/external zones
                cmap_norm = colors.LogNorm(
                    filtered["matrix_forecast"].min(), filtered["matrix_forecast"].max()
                )
                lower, upper = np.inf, -np.inf
                for m, ie in (("o", "Internal"), ("+", "External")):
                    ax = axd[ie.lower()]

                    mask = filtered.index.get_level_values("IE").str.lower() == ie.lower()
                    scatter = ax.scatter(
                        filtered.loc[mask, "matrix"],
                        filtered.loc[mask, "tempro"],
                        marker=m,
                        label=f"Growth Factors\n{ie} Zones",
                        c=filtered.loc[mask, "matrix_forecast"],
                        cmap="YlGn",
                        norm=cmap_norm,
                    )
                    _linear_fit(filtered.loc[mask], ax, "black", "Linear Fit")

                    ax.set_aspect("equal")
                    ax.legend()
                    ax.set_xlabel("Model Growth Factors")
                    ax.set_ylabel("TEMPro Growth Factors")
                    ax.set_title(f"{ie} Zones")

                    # Keep track of min/max axis bounds
                    for i in ("x", "y"):
                        bounds = getattr(ax, f"get_{i}lim")()
                        if bounds[0] < lower:
                            lower = bounds[0]
                        if bounds[1] > upper:
                            upper = bounds[1]

                # Set consistent axis bounds
                for ie in ("internal", "external"):
                    axd[ie].set_xlim(lower, upper)
                    axd[ie].set_ylim(lower, upper)

                fig.suptitle(f"{title}\n{to.get_name().upper()} {pa.title()}")
                cbar = fig.colorbar(
                    scatter,
                    label=f"Model {to.get_name().upper()} {pa.title()}",
                    cax=axd["colorbar"],
                )
                # cbar.ax.yaxis.set_minor_formatter(
                #     ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(5, 1.5))
                # )

                pdf.savefig(fig)
                plt.close()

    LOG.info("Written: %s", output_path)


def ntem_tempro_comparison_plots(
    comparison_folder: Path,
    geospatial_file: GeoSpatialFile,
    plot_zoning: str,
    analytical_area_shape: GeoSpatialFile,
    output_folder: Path,
):
    """Create growth comparison maps and CSVs for NTEM vs TEMPro.

    Parameters
    ----------
    comparison_folder : Path
        Folder containing growth comparison data CSVs.
    geospatial_file : GeoSpatialFile
        Polygon shapefile containing data for mapping.
    plot_zoning : str
        Level of zoning to do the plot at.
    analytical_area_shape : GeoSpatialFile
        Shapefile showing the analytical area, used as a boundary
        on the map.
    output_folder : Path
        Folder to save output CSVs and PDF graphs to.
    """
    LOG.info("Producing NTEM vs TEMPro comparison plots")
    nice_zone_name = {"lad_2020_internal_noham": "LAD Internal NoHAM External"}
    geospatial = get_geo_data(geospatial_file).to_frame()
    analytical_area = get_geo_data(analytical_area_shape).iloc[0]

    plot_zone_system = nd.get_zoning_system(plot_zoning)
    geospatial.loc[:, "IE"] = np.nan
    geospatial.loc[geospatial.index.isin(plot_zone_system.internal_zones), "IE"] = "Internal"
    geospatial.loc[geospatial.index.isin(plot_zone_system.external_zones), "IE"] = "External"
    geospatial.dropna(axis=0, inplace=True)

    for year, file in _tempro_comparisons_iterator(comparison_folder, plot_zoning):
        columns = {
            "matrix_type": "trip_origin",
            "trip_end_type": "pa",
            "p": "p",
            f"{plot_zoning}_zone_id": "zone",
            "matrix_2018": "matrix_base",
            f"matrix_{year}": "matrix_forecast",
            "tempro_2018": "tempro_base",
            f"tempro_{year}": "tempro_forecast",
            "matrix_growth": "matrix_growth",
            "tempro_growth": "tempro_growth",
            "growth_difference": "growth_difference",
        }
        comparison = pd.read_csv(file, usecols=columns.keys()).rename(columns=columns)
        comparison = comparison.merge(
            geospatial, left_on="zone", right_index=True, how="left", validate="m:1"
        )

        # Group into HB/NHB productions and attractions, then recalculate growth
        trip_end_groups = comparison.groupby(
            ["zone", "trip_origin", "pa", "IE"], as_index=False
        ).agg(
            {
                "matrix_base": "sum",
                "matrix_forecast": "sum",
                "tempro_base": "sum",
                "tempro_forecast": "sum",
                "geometry": "first",
            }
        )
        for nm in ("matrix", "tempro"):
            trip_end_groups.loc[:, f"{nm}_growth"] = (
                trip_end_groups[f"{nm}_forecast"] / trip_end_groups[f"{nm}_base"]
            )
        trip_end_groups.loc[:, "growth_difference"] = (
            trip_end_groups["matrix_growth"] - trip_end_groups["tempro_growth"]
        )
        plot_iterator = (
            trip_end_groups[["trip_origin", "pa"]].drop_duplicates().itertuples(index=False)
        )
        trip_end_groups.set_index(["zone", "trip_origin", "pa", "IE"], inplace=True)
        out = output_folder / f"PA_TEMPro_growth_comparison_{year}_{plot_zoning}.csv"
        trip_end_groups.drop(columns=["geometry"]).to_csv(out)
        LOG.info("Written: %s", out)

        growth_comparison_regression(
            trip_end_groups,
            out.with_name(out.stem + "-scatter.pdf"),
            "NTEM Model and TEMPro Trip End Growth Comparison at {}".format(
                nice_zone_name.get(plot_zoning, plot_zoning)
            ),
        )

        plot_column = "Growth Difference"
        trip_end_groups = gpd.GeoDataFrame(
            trip_end_groups, crs=geospatial.crs, geometry="geometry"
        )
        trip_end_groups.rename(columns={"growth_difference": plot_column}, inplace=True)
        out = out.with_suffix(".pdf")
        with backend_pdf.PdfPages(out) as pdf:
            # Calculate consistent bins for all heatmaps
            neg_bins = mapclassify.NaturalBreaks(
                trip_end_groups.loc[trip_end_groups[plot_column] <= 0, plot_column], k=5
            )
            pos_bins = mapclassify.NaturalBreaks(
                trip_end_groups.loc[trip_end_groups[plot_column] >= 0, plot_column], k=5
            )
            bins = np.concatenate([neg_bins.bins, [0], pos_bins.bins])

            for to, pa in plot_iterator:
                fig = _heatmap_figure(
                    trip_end_groups.loc[:, to, pa],
                    plot_column,
                    f"{to.upper()} {pa.title()} NTEM & TEMPro Growth Comparison"
                    f"\n at {nice_zone_name.get(plot_zoning, plot_zoning)}",
                    bins=bins,
                    analytical_area=analytical_area,
                    positive_negative_colormaps=True,
                )
                pdf.savefig(fig)
                plt.close()

        LOG.info("Written: %s", out)


def add_analytical_area(
    ax: plt.Axes, area: Union[geometry.MultiPolygon, geometry.Polygon]
) -> patches.Polygon:
    """Add analytical area boundary to a map.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add boundary to.
    area : Union[geometry.MultiPolygon, geometry.Polygon]
        Analytical area polygon of boundary to add.

    Returns
    -------
    patches.Polygon
        The patch added to the axes.

    Raises
    ------
    TypeError
        If `area` isn't a Polygon or MultiPolygon.
    """
    if isinstance(area, geometry.Polygon):
        polygons = geometry.MultiPolygon([area])
    elif isinstance(area, geometry.MultiPolygon):
        polygons = area
    else:
        raise TypeError(f"unexpected type ({type(area)}) for area")

    legend_patch = None
    for i, poly in enumerate(polygons.geoms):
        patch = patches.Polygon(
            poly.exterior.coords,
            ec="red",
            fill=False,
            linewidth=2,
            label="North Analytical\nArea Boundary" if i == 0 else None,
            zorder=2,
        )
        if i == 0:
            legend_patch = patch
        ax.add_patch(patch)
    return legend_patch


def _tempro_comparison_matrix_growth(
    excel: Path, base_year: int, forecast_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare matrix and TEMPro growth and return summaries."""
    GROUP_COLS = ["matrix_type", "purpose"]

    tempro: pd.DataFrame = pd.read_excel(
        excel,
        sheet_name="TEMPro Data",
        usecols=["matrix_type", "p", f"tempro_{base_year}", f"tempro_{forecast_year}"],
    )
    tempro.rename(columns={"p": "purpose"}, inplace=True)
    tempro = tempro.groupby(GROUP_COLS).sum()
    tempro.loc[:, "tempro_growth"] = (
        tempro[f"tempro_{forecast_year}"] / tempro[f"tempro_{base_year}"]
    )

    matrices = pd.read_excel(
        excel,
        sheet_name=["Base Matrices Data", "Forecast Matrices Data"],
        usecols=["matrix_type", "purpose", "trips"],
    )
    matrices: Dict[str, pd.DataFrame] = {k.split()[0].lower(): v for k, v in matrices.items()}
    for nm, mat in matrices.items():
        mat = mat.groupby(GROUP_COLS).sum()
        matrices[nm] = mat.rename(columns={"trips": nm})

    combined = pd.concat([tempro, matrices["base"], matrices["forecast"]], axis=1)
    combined.reset_index(inplace=True)

    combined.loc[:, "matrix_growth"] = combined["forecast"] / combined["base"]
    combined.loc[:, "matrix_type"] = combined["matrix_type"].str.upper()
    combined.loc[:, "growth_difference"] = (
        combined["matrix_growth"] - combined["tempro_growth"]
    )
    combined.columns = combined.columns.str.replace("_", " ").str.title()

    growth = combined.loc[
        :, ["Matrix Type", "Purpose", "Matrix Growth", "Tempro Growth", "Growth Difference"]
    ].copy()
    growth.insert(2, "Year", forecast_year)

    return combined, growth


def _tempro_comparisons_iterator(
    comparisons_folder: Path, zoning: str, file_type: Optional[str] = None
) -> Iterator[Tuple[int, Path]]:
    """Iterate through PA TEMPro comparison spreadsheets."""
    file_name = f"PA_TEMPro_comparisons-{{year}}-{zoning}"
    if file_type is None:
        file_type = ".*"

    for path in comparisons_folder.glob(file_name.format(year="????") + file_type):
        match = re.match(
            file_name.format(year=r"(\d{4})"),
            file_ops.remove_suffixes(path).stem,
            re.IGNORECASE,
        )
        if match is None:
            LOG.warning("Skipping file %s because cannot find year in name", path.name)
            continue

        year = int(match.group(1))
        yield year, path


def tempro_comparison_summary(comparisons_folder: Path, zoning: str, base_year: int) -> None:
    """Produce matrix totals TEMPro comparisons summary by purpose.

    Parameters
    ----------
    comparisons_folder : Path
        Folder containing the TEMPro comparisons spreadsheets.
    zoning : str
        Name of the matrix zoning system to read the TEMPro comparisons
        for, the output file contains matrix totals.
    base_year : int
        Base model year.
    """
    output_path = comparisons_folder / "PA_TEMPro_comparisons_summary.xlsx"
    LOG.info("Summarising TEMPro comparisons")

    with pd.ExcelWriter(output_path) as excel_file:
        growth_dfs = []
        for year, path in _tempro_comparisons_iterator(comparisons_folder, zoning, ".xlsx"):
            LOG.info("Summarising: %s", path.name)
            comparison, growth = _tempro_comparison_matrix_growth(path, base_year, year)
            comparison.to_excel(excel_file, sheet_name=str(year), index=False)
            growth_dfs.append(growth)

        growth = pd.concat(growth_dfs)
        growth.to_excel(excel_file, sheet_name="Growth Summary", index=False)

    LOG.info("Written: %s", output_path)


def tempro_uc_summary(comparisons_folder: Path, zoning: str) -> None:
    """Convert PA TEMPro comparison spreadsheets from purpose to user class.

    Parameters
    ----------
    comparisons_folder : Path
        Folder containing TEMPro comparison spreadsheets.
    zoning : str
        Name of the matrix zoning system for the TEMPro comparisons.
    """
    purpose_lookup = {
        1: "commute",
        **dict.fromkeys((2, 12), "business"),
        **dict.fromkeys(itertools.chain(range(3, 9), range(13, 17), (18,)), "other"),
    }

    LOG.info("Summarising TEMPro comparisons by user class")
    for _, excel_path in _tempro_comparisons_iterator(comparisons_folder, zoning, ".xlsx"):
        output_path = excel_path.with_name(excel_path.stem + "-by_userclass.xlsx")

        # Copy file and update new version
        shutil.copy(excel_path, output_path)

        with pd.ExcelWriter(output_path, mode="a", if_sheet_exists="replace") as excel_out:
            wb: openpyxl.Workbook = excel_out.book

            summary = wb["Summary"]
            # Clear data validations before adding UC validation,
            # otherwise new validation won't overwrite old
            summary.data_validations = DataValidationList()
            options = tuple(set(purpose_lookup.values()))
            valid_uc = DataValidation(type="list", formula1=f'"{",".join(options)}"')
            summary.add_data_validation(valid_uc)

            purp_cell = "C2"
            valid_uc.add(purp_cell)
            summary["C2"] = options[0]

            sheet_name = "TEMPro Data"
            tempro = pd.read_excel(excel_out, sheet_name=sheet_name)
            tempro.loc[:, "p"] = tempro["p"].replace(purpose_lookup)
            tempro.rename(columns={"p": "uc"}, inplace=True)

            tempro.loc[:, "id"] = (
                tempro["trip_end_type"]
                + "_"
                + tempro["uc"]
                + "_"
                + tempro[f"{zoning}_zone_id"].astype(str)
            )

            tempro = tempro.groupby(
                ["matrix_type", "trip_end_type", "uc", "m", f"{zoning}_zone_id", "id"],
                as_index=False,
            ).sum()
            tempro.to_excel(excel_out, sheet_name=sheet_name, index=False)

            for nm in ("Base", "Forecast"):
                sheet_name = f"{nm} Matrices Data"
                df = pd.read_excel(excel_out, sheet_name=sheet_name)
                df.loc[:, "purpose"] = df["purpose"].replace(purpose_lookup)
                df.rename(columns={"purpose": "uc"}, inplace=True)

                df.loc[:, "id"] = (
                    df["uc"]
                    + "_"
                    + df["from_zone"].astype(str)
                    + "_"
                    + df["to_zone"].astype(str)
                )

                df = df.groupby(
                    ["matrix_type", "uc", "from_zone", "to_zone", "id"], as_index=False
                ).sum()
                df.to_excel(excel_out, sheet_name=sheet_name, index=False)

        LOG.info("Written: %s", output_path)


def main(params: PAPlotsParameters) -> None:
    """Produce the PA growth and TEMPro comparison maps and graphs.

    Parameters
    ----------
    params : PAPlotsParameters
        Parameters and input files for creating the graphs.
    """
    params.output_folder.mkdir(exist_ok=True)
    tempro_comparison_summary(
        params.tempro_comparison_folder,
        params.tempro_comparison_summary_zoning,
        params.base_year,
    )
    tempro_uc_summary(params.tempro_comparison_folder, params.tempro_comparison_summary_zoning)
    ntem_pa_plots(
        params.base_matrix_folder,
        params.forecast_matrix_folder,
        params.matrix_zoning,
        params.plot_zoning,
        params.output_folder,
        params.geospatial_file,
        params.analytical_area_shape,
    )
    ntem_tempro_comparison_plots(
        params.tempro_comparison_folder,
        params.geospatial_file,
        params.plot_zoning,
        params.analytical_area_shape,
        params.output_folder,
    )


def _colormap_classify(
    data: pd.Series,
    cmap_name: str,
    n_bins: int = 5,
    label_fmt: str = "{:.0f}",
    bins: Optional[List[Union[int, float]]] = None,
) -> CustomCmap:
    """Calculate a NaturalBreaks colour map."""

    def make_label(lower: float, upper: float) -> str:
        if lower == -np.inf:
            return "< " + label_fmt.format(upper)
        if upper == np.inf:
            return "> " + label_fmt.format(lower)
        return label_fmt.format(lower) + " - " + label_fmt.format(upper)

    finite = data.dropna()
    if finite.empty:
        # Return empty colour map
        return CustomCmap(pd.Series(dtype=float), pd.DataFrame(dtype=float), [])

    if bins is not None:
        mc_bins = mapclassify.UserDefined(finite, bins)
    else:
        mc_bins = mapclassify.NaturalBreaks(finite, k=n_bins)

    bin_categories = pd.Series(mc_bins.yb, index=finite.index)
    bin_categories = bin_categories.reindex_like(data)

    cmap = cm.get_cmap(cmap_name, mc_bins.k)
    colours = pd.DataFrame(
        cmap(bin_categories), index=bin_categories.index, columns=iter("RGBA")
    )
    colours.loc[bin_categories.isna(), :] = np.nan

    min_bin = np.min(finite)
    if min_bin > mc_bins.bins[0]:
        if mc_bins.bins[0] > 0:
            min_bin = 0
        else:
            min_bin = -np.inf

    bins = [min_bin, *mc_bins.bins]
    labels = [make_label(l, u) for l, u in zip(bins[:-1], bins[1:])]
    legend = [
        patches.Patch(fc=c, label=l, ls="") for c, l in zip(cmap(range(mc_bins.k)), labels)
    ]

    return CustomCmap(bin_categories, colours, legend)


def _autoplot_func(
    years: list,
    df: pd.DataFrame,
    zone_shape: gpd.GeoDataFrame,
    out_folder: Path,
    hb_nhb: str,
    pa: str,
    iter: str,
    level: str = None,
    pct=False,
    diff=False,
    plot_type="heatmap",
):
    for i in range(len(years) - 1):
        if pct:
            growth = (df[years[i + 1]] / df[years[i]]) - 1
        elif diff:
            growth = df[years[i + 1]] - df[years[i]]
        with backend_pdf.PdfPages(
            os.path.join(out_folder, f'iter{iter}', 'NTEM', f'{hb_nhb}_{pa}','reports', f"pct_plot_{years[i]}_{level}.pdf")
        ) as pdf:
            if plot_type == "heatmap":
                cols = growth.columns
                to_plot = zone_shape.join(growth)
                for column in cols:
                    fig = _heatmap_figure(
                        to_plot,
                        column_name=column,
                        title=f"{years[i]}_{column}",
                        positive_negative_colormaps=True,
                        n_bins=10,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)


def switch_keys(dic: dict) -> dict:
    """
    Short function to switch keys in a nested dictionary
    Args:
    The dictionary you want keys switched for
    Returns:
    The same dictionary with keys switched
    """
    d = collections.defaultdict(dict)
    for key, value in dic.items():
        for k, v in value.items():
            d[k][key] = v
    return d


def plot_tripend_iters(
    scenario: str,
    pa: str,
    hb_nhb: str,
    zone_name: str,
    segmentation: str,
    years: list = None,
    pct=False,
    diff=False,
    plot_type: str = "heatmap",
    ind_segs=True,
):
    """
    Function for outputting a range of plots for a given trip-end dvector, or set of dvectors for different years
    Args:
        Dvec_path (str): This should be a path to the dvector you want plotted.  If you are providing just one, give
        the full path, if you want multiple years compared, replace the year with {}
        zoning (str): the zoning system you want the data plotted at 
        segmentation (str): The segmenation you want the data plotted at.  Can be either plotted at every segmentation, or grouped by each level.
        i.e., could be grouped by m, then each mode plotted, then grouped by tp and each tp plotted
        years (list, optional): If you want year on year comparisons, supply a list of the years to compare. Defaults to None.
        pct (bool, optional): If you have supplied a list of years, set this to True if you want the percentage growth plotted. Defaults to False.
        diff (bool, optional): If you have supplied a list of years, set this to True if you want the absolute growth plotted. Defaults to False.
        plot_type (str, optional): Currently only heatmap, will be updated for different plot types. Defaults to 'heatmap'.
        ind_segs (bool, optional): If True, every segment will be plotted (i.e. (mode1, tp1)).  If False data will be grouped by each segment level
        (e.g. mode 1, mode 2... tp 1, tp 2...)
    """
    path = Path(r"T:\MidMITs Demand\MiTEM")
    seg = nd.get_segmentation_level(segmentation)
    zone = nd.get_zoning_system(zone_name)
    metadata = zone.get_metadata()
    zone_index = zone_name + "_zone_id"
    segments = seg.naming_order
    index = [zone_index] + segments
    zone_shape = (
        gpd.read_file(metadata.shapefile_path)[[metadata.shapefile_id_col, "geometry"]]
        .rename(columns={metadata.shapefile_id_col: f"{zone_name}_zone_id"})
        .set_index(f"{zone_name}_zone_id")
    )
    if years:
        if ind_segs:
            data = {}
            for year in years:
                dvec = (
                    nd.DVector.load(
                        os.path.join(
                            path,
                            f'iter{scenario}',
                            'NTEM',
                            f'{hb_nhb}_{pa}',
                            f'{hb_nhb}_msoa_notem_segmented_{year}_dvec.pkl',
                        )
                    )
                    .translate_zoning(zone)
                    .aggregate(seg)
                )
                year_data = {}
                for level in segments:
                    drop = segments.copy()
                    drop.remove(level)
                    df = (
                        dvec.to_df()
                        .groupby([zone_name + "_zone_id", level])
                        .sum()
                        .drop(drop, axis=1)
                        .unstack(level)
                        .rename(columns={"val": year}, level=0)
                    )
                    year_data[level] = df
                data[year] = year_data

            d = switch_keys(data)

            for level in segments:
                df = ft.reduce(
                    lambda left, right: pd.merge(
                        left, right, right_index=True, left_index=True
                    ),
                    d[level].values(),
                )
                _autoplot_func(
                    years=years,
                    zone_shape=zone_shape,
                    out_folder=os.path.join(path),
                    pct=pct,
                    diff=diff,
                    df=df,
                    level=level,
                    plot_type=plot_type,
                    hb_nhb=hb_nhb,
                    pa=pa,
                    iter=scenario
                )
        else:
            data = {}
            for year in years:
                dvec = (
                    nd.DVector.load(
                        os.path.join(
                            path,
                            scenario,
                            "NTEM",
                            f"{hb_nhb}_{pa}",
                            f"{hb_nhb}_msoa_notem_segmented_{year}_dvec.pkl",
                        )
                    )
                    .aggregate(seg)
                    .translate_zoning(zone)
                )
                df = (
                    dvec.to_df()
                    .set_index(index)
                    .unstack(segments)
                    .rename(columns={"val": year}, level=0)
                )
                data[year] = df
            df = ft.reduce(
                lambda left, right: pd.merge(left, right, right_index=True, left_index=True),
                data.values(),
            )
            _autoplot_func(
            years=years,
            zone_shape=zone_shape,
            out_folder=os.path.join(path),
            pct=pct,
            diff=diff,
            df=df,
            level=level,
            plot_type=plot_type,
            hb_nhb=hb_nhb,
            pa=pa,
            iter=scenario,
            )


##### MAIN #####
if __name__ == "__main__":
    iteration_folder = Path(r"I:\NorMITs Demand\noham\NTEM\iter1d")
    forecast_matrix_folder = iteration_folder / r"Matrices\24hr VDM PA Matrices"

    pa_parameters = PAPlotsParameters(
        base_matrix_folder=Path(r"I:\NorMITs Demand\import\noham\post_me\tms_seg_pa"),
        forecast_matrix_folder=forecast_matrix_folder,
        matrix_zoning="noham",
        plot_zoning="lad_2020_internal_noham",
        output_folder=forecast_matrix_folder / "Plots",
        geospatial_file=GeoSpatialFile(
            Path(
                r"Y:\Data Strategy\GIS Shapefiles"
                r"\lad_2020_internal_noham\lad_2020_internal_noham_zoning.shp"
            ),
            "zone_name",
        ),
        analytical_area_shape=GeoSpatialFile(
            Path(
                r"Y:\Data Strategy\GIS Shapefiles\North Analytical Area"
                r"\Boundary\north_analytical_area_simple_boundary.shp"
            ),
            "Name",
        ),
        tempro_comparison_folder=iteration_folder / r"Matrices\PA\TEMPro Comparisons",
        tempro_comparison_summary_zoning="3_sector",
        base_year=2018,
    )

    main(pa_parameters)
