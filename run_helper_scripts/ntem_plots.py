# -*- coding: utf-8 -*-
"""
    Script for producing maps and graphs for the NTEM forecasting report.
"""

##### IMPORTS #####
from __future__ import annotations

# Standard imports
import collections
import dataclasses
import enum
from pathlib import Path
import re
import sys
from typing import Any, Iterator, NamedTuple, Union
import warnings

# Third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, figure, ticker
import matplotlib.backends.backend_pdf as backend_pdf
from shapely import geometry


# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import file_ops, pandas_utils
from normits_demand.reports import ntem_forecast_checks
from normits_demand import colours as tfn_colours
from normits_demand.core import enumerations as nd_enum


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
    productions: pd.DataFrame
    attractions: pd.DataFrame

    def _check_other(self, other: Any, operation: str) -> None:
        if not isinstance(other, MatrixTripEnds):
            raise TypeError(f"cannot perform {operation} with {type(other)}")

    def __add__(self, other: MatrixTripEnds) -> MatrixTripEnds:
        self._check_other(other, "addition")
        return MatrixTripEnds(
            self.productions + other.productions, self.attractions + other.attractions
        )

    def __truediv__(self, other: MatrixTripEnds) -> MatrixTripEnds:
        self._check_other(other, "division")
        return MatrixTripEnds(
            self.productions / other.productions, self.attractions / other.attractions
        )


class GeoSpatialFile(NamedTuple):
    path: Path
    id_column: str


@dataclasses.dataclass
class PAPlotsParameters:
    base_matrix_folder: Path
    forecast_matrix_folder: Path
    matrix_zoning: str
    plot_zoning: str
    output_folder: Path
    geospatial_file: GeoSpatialFile
    analytical_area_shape: Path


class PlotType(nd_enum.AutoName):
    BAR = enum.auto()
    LINE = enum.auto()


##### FUNCTIONS #####
def match_files(folder: Path, pattern: re.Pattern) -> Iterator[tuple[dict[str, str], Path]]:
    for file in folder.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.stem)
        if match is None:
            continue
        # FIXME Remove if statement
        md = match.groupdict()
        if md["trip_origin"] == "hb":
            if ("user_class" in md and md["user_class"] == "commute") or (
                "purpose" in md and md["purpose"] == "1"
            ):
                yield match.groupdict(), file
                break


def matrix_totals(
    matrix: pd.DataFrame, zoning_name: str, trip_origin: str, user_class: str, year: str
) -> pd.DataFrame:
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
    mat = file_ops.read_df(matrix, find_similar=True, index_col=0)
    mat.columns = pd.to_numeric(mat.columns, downcast="integer")
    totals = matrix_totals(mat, matrix_zoning, **kwargs)
    if to_zoning:
        mat = ntem_forecast_checks.translate_matrix(mat, matrix_zoning, to_zoning)
        matrix_zoning = to_zoning
    return MatrixTripEnds(mat.sum(axis=1), mat.sum(axis=0)), totals


def get_base_trip_ends(
    folder: Path, matrix_zoning: str, plot_zoning: str
) -> tuple[dict[tuple[str, str], MatrixTripEnds], pd.DataFrame]:
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


def plot_bars(
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
    bars = ax.bar(
        x_data,
        y_data,
        label=label,
        color=colour,
        width=width,
        align="edge",
    )

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


def plot_line(
    ax: plt.Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    colour: str,
    label: str,
    label_fmt: str = ".3g",
):
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
                colour=colours[uc],
                label=f"{to.upper()} - {uc.title()}",
                label_fmt=fmt,
            )
            if plot_type == PlotType.BAR:
                plot_bars(
                    ax, x + i * width, row.values, width=width, max_height=max_height, **kwargs
                )
            elif plot_type == PlotType.LINE:
                years = pd.to_numeric(row.index, downcast="integer")
                plot_line(ax, years, row.values, **kwargs)
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
    totals: pd.DataFrame,
    out_path: Path,
):
    # TODO Plot internal and external separately
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
                curr_tot.div(curr_tot["2018"], axis=0).drop(columns="2018"),
                f"NTEM Forecasting PA - {area.title()} Matrix Growth",
                "Matrix Growth",
                plot_type=PlotType.LINE,
            )
            pdf.savefig(growth_figure)

    LOG.info("Written: %s", out_path)


def get_geo_data(file: GeoSpatialFile) -> gpd.GeoSeries:
    geo = gpd.read_file(file.path)
    if file.id_column not in geo.columns:
        raise KeyError(f"{file.id_column} missing from {file.path.name}")
    geo = geo.set_index(file.id_column)
    return geo["geometry"]


def _heatmap_figure(
    geodata: gpd.GeoDataFrame, column_name: str, title: str, bins: list[int] = None
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 15), frameon=False, constrained_layout=True)
    fig.suptitle(title, fontsize="xx-large")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_axis_off()

    # Drop nan
    geodata = geodata.loc[~geodata[column_name].isna()]

    kwargs = dict(
        column=column_name,
        cmap="viridis_r",
        scheme="NaturalBreaks",
        k=7,
        legend_kwds=dict(
            title=f"{column_name.title()}",
            title_fontsize="xx-large",
            fontsize="x-large",
            fmt="{:.3g}",
        ),
        # missing_kwds={
        #     "color": "lightgrey",
        #     "edgecolor": "red",
        #     "hatch": "///",
        #     "label": "Missing values",
        # },
        linewidth=0.001,
    )
    if bins:
        kwargs["scheme"] = "UserDefined"
        kwargs["classification_kwds"] = {"bins": bins}
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

    axes[1].set_xlim(290000, 560000)
    axes[1].set_ylim(300000, 680000)
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
    bins: list[int] = None,
):
    growth = forecast / base

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

            fig = _heatmap_figure(geodata, name, title, bins)
            pdf.savefig(fig)

    LOG.info("Saved: %s", output_path)


def ntem_pa_plots(
    base_folder: Path,
    forecast_folder: Path,
    matrix_zoning: str,
    plot_zoning: str,
    out_folder: Path,
    geospatial_file: GeoSpatialFile,
):
    warnings.filterwarnings(
        "ignore", message=".*zones in the matrix are missing", category=UserWarning
    )
    base_trip_ends, base_totals = get_base_trip_ends(base_folder, matrix_zoning, plot_zoning)
    index_cols = ["matrix_area", "trip_origin", "user_class"]
    base_totals = base_totals.groupby(index_cols).sum()

    geospatial = get_geo_data(geospatial_file)

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

        # TODO Create maps of trip end growth
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
                params["year"],
                params["trip_origin"].upper(),
                params["user_class"].title(),
            ),
        )
        print()

    forecast_totals: pd.DataFrame = pd.concat(forecast_totals)
    forecast_totals = forecast_totals.groupby(index_cols).agg(np.nansum)
    plot_all_matrix_totals(
        pd.concat([base_totals, forecast_totals], axis=1),
        out_folder / "NTEM_PA_matrix_totals.pdf",
    )

def remove_poly_holes(polygon: Union[geometry.Polygon, geometry.MultiPolygon]) -> Union[geometry.Polygon, geometry.MultiPolygon]:
    if isinstance(polygon, geometry.Polygon):
        return geometry.Polygon(polygon.exterior.coords)
    elif not isinstance(polygon, geometry.MultiPolygon):
        raise TypeError(f"expected Polygon or MultiPolygon not {type(polygon)}")
    
    multi = []
    for poly in polygon.geoms:
        multi.append(geometry.Polygon(poly))
    return geometry.MultiPolygon(multi)


def ntem_tempro_comparison_plots(
    comparison_folder: Path, geospatial_file: GeoSpatialFile, plot_zoning: str, analytical_area_shape: Path
):
    comp_zone_lookup = {"lad_2020": "LAD"}
    geospatial = get_geo_data(geospatial_file)
    analytical_area = gpd.read_file(analytical_area_shape)
    analytical_area = analytical_area.to_crs(epsg=27700)
    buff = analytical_area.copy()
    buff.loc[:, "geometry"] = analytical_area.buffer(100)
    buff.plot(ec="red", linewidth=2)
    plt.show()
    dissolve = analytical_area.dissolve()
    dissolve.loc[:, "geometry"] = dissolve.geometry.apply(remove_poly_holes)
    geom = dissolve.loc[:, "geometry"]
    # TODO Add TfN area overlay to maps
    geoms = gpd.GeoDataFrame(geom,)
    dissolve.plot(ec="red", linewidth=2)
    plt.show()


    for geo in dissolve.iloc[0, 0]:
        print(geo.area / 1e6)

    comparison_filename = "PA_TEMPro_comparisons-{year}-{zone}"
    zone = comp_zone_lookup.get(plot_zoning, plot_zoning)
    for file in comparison_folder.glob(comparison_filename.format(year="*", zone=zone) + "*"):
        match = re.match(comparison_filename.format(year=r"(\d+)", zone=zone), file.stem)
        if not match or file.is_dir():
            continue
        year = int(match.group(1))

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
        comparison = comparison.merge(geospatial, left_on="zone", right_index=True, how="left", validate="m:1")
        
        # Group into HB/NHB productions and attractions, then recalculate growth
        trip_end_groups = comparison.groupby(["zone", "trip_origin", "pa"], as_index=False).agg({"matrix_base": "sum", "matrix_forecast": "sum"
        , "tempro_base": "sum", "tempro_forecast": "sum", "geometry": "first"})
        for nm in ("matrix", "tempro"):
            trip_end_groups.loc[:, f"{nm}_growth"] = trip_end_groups[f"{nm}_forecast"] / trip_end_groups[f"{nm}_base"]
        trip_end_groups.loc[:, "growth_difference"] = trip_end_groups["matrix_growth"] - trip_end_groups["tempro_growth"]

        for to, pa in trip_end_groups[["trip_origin", "pa"]].drop_duplicates().itertuples(index=False):
            print(to, pa)


def main(params: PAPlotsParameters) -> None:
    params.output_folder.mkdir(exist_ok=True)
    # ntem_pa_plots(
    #     params.base_matrix_folder,
    #     params.forecast_matrix_folder,
    #     params.matrix_zoning,
    #     params.plot_zoning,
    #     params.output_folder,
    #     params.geospatial_file,
    # )
    ntem_tempro_comparison_plots()
    print()


##### MAIN #####
if __name__ == "__main__":
    # forecast_matrix_folder=Path(
    #     r"I:\NorMITs Demand\noham\NTEM\iter1c\Matrices\24hr VDM PA Matrices"
    # )
    forecast_matrix_folder = Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Outputs\NTEM\iter1c\Matrices\24hr VDM PA Matrices"
    )

    pa_parameters = PAPlotsParameters(
        # base_matrix_folder=Path(r"I:\NorMITs Demand\import\noham\post_me\tms_seg_pa"),
        base_matrix_folder=Path(
            r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Inputs\noham\post_me\tms_seg_pa"
        ),
        forecast_matrix_folder=forecast_matrix_folder,
        matrix_zoning="noham",
        plot_zoning="lad_2020",
        output_folder=forecast_matrix_folder / "Plots",
        geospatial_file=GeoSpatialFile(
            Path(
                r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\GIS"
                r"\Local_Authority_Districts_(December_2020)_UK_BFC"
                r"\Local_Authority_Districts_(December_2020)_UK_BGC.shp"
            ),
            "LAD20CD",
        ),
        tempro_comparison_folder=Path(
            r"I:\NorMITs Demand\noham\NTEM\iter1c\Matrices\PA\TEMPro Comparisons"
        ),
        analytical_area_shape=Path(r"Y:\Data Strategy\GIS Shapefiles\North Analytical Area\north_analytical_area.shp"),
    )

    main(pa_parameters)
