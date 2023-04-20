# -*- coding: utf-8 -*-
"""Script for creating yearly comparisons for EDDIE inputs and DDGs.

Note: This script is not very flexible and is written to create
specific plots using outputs from the EDDIE adjustments processes.
"""

##### IMPORTS #####
# Standard imports
import logging
import pathlib
import re
from typing import Optional

# Third party imports
from matplotlib import pyplot as plt, ticker
from matplotlib.backends import backend_pdf
import numpy as np
import pandas as pd

# Local imports

##### CONSTANTS #####
LOG = logging.getLogger(__name__)
NORTH_REGIONS = [
    "north east",
    "north west",
    "yorkshire & the humber",
]
OTHER_GB_REGIONS = [
    "wales",
    "scotland",
    "west midlands",
    "east midlands",
    "east of england",
    "south east",
    "south west",
    "london",
    # "northern ireland",
]
BASE_YEAR = 2018

##### CLASSES #####

##### FUNCTIONS #####
def moving_average(x: np.ndarray, n: int = 4) -> np.ndarray:
    """Calculating moving average on array (`x`) with period `n`."""
    return np.convolve(x, np.ones(n), "valid") / n


def _plot_yearly_comparisons(
    data: dict[str, pd.DataFrame],
    fig_title: str,
    popemp: str,
    regions: list[str],
    quarterly: bool,
    plot_titles: dict[str, str],
    yaxis_labels: dict[str, str],
    npier_scenario: str = "NPIER",
    region_replace: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """Create line / scatter plots for the region totals.
    
    Internal function used by `_regions_plot_internal`.
    """
    fig, axes = plt.subplots(3, figsize=(10, 15), constrained_layout=True, sharex=True)
    fig.suptitle(fig_title, fontsize="xx-large", weight="medium")

    for i, (nm, df) in enumerate(data.items()):
        if region_replace is not None:
            df.index = df.index.to_series().replace(region_replace)
        ax = axes[i]

        if quarterly:
            # calculate the positions of the borders between the years
            pos = []
            years = []
            for j, column in enumerate(df.columns):
                year, quarter = column.split("_")
                year = int(year)

                if quarter.upper() == "Q1":
                    pos.append(j)
                    years.append(year)

            pos.append(len(df.columns))
            pos = np.array(pos) - 0.5

        for region_i, region in enumerate(regions):
            if not quarterly or nm in (npier_scenario, "EDDIE"):
                ax.plot(df.columns, df.loc[region, :], label=region.title())
            else:
                ax.scatter(df.columns, df.loc[region, :], marker="+", s=3, c=f"C{region_i}")
                finite = df.loc[region, :].dropna()
                average = moving_average(finite.values)

                ax.plot(
                    finite.index[len(finite) - len(average) :],
                    average,
                    label=f"{region.title()} Rolling Mean",
                    c=f"C{region_i}",
                    ls="--",
                )

        if nm in (npier_scenario, "EDDIE"):
            ax.yaxis.set_major_formatter("{x:.2g}")
        else:
            abs_max = np.nanmax(df.loc[regions].abs().values)
            decimals = max(abs(np.floor(np.log10(abs_max))) - 1, 0)

            if decimals < 4:
                fmt = "{x:.{sf}%}".replace("{sf}", f"{decimals:.0f}")
                ax.yaxis.set_major_formatter(fmt)
            else:
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{x * 100:.2g}%")
                )

        if quarterly:
            ax.xaxis.set_major_locator(ticker.FixedLocator(pos))
            ax.xaxis.set_minor_locator(ticker.IndexLocator(1, 0))
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.tick_params(axis="x", length=10)

            # years at the center of their range
            for year, pos0, pos1 in zip(years, pos[:-1], pos[1:]):
                ax.text(
                    (pos0 + pos1) / 2,
                    -0.07,
                    year,
                    ha="center",
                    clip_on=False,
                    transform=ax.get_xaxis_transform(),
                    fontdict=dict(fontsize="x-small"),
                    rotation=75,
                )
        else:
            ax.tick_params(axis="x", labelrotation=75, labelsize="small")

        ax.grid(True, which="major", alpha=0.2)
        ax.tick_params(labelsize="small")
        ax.set_title(plot_titles[nm].format(popemp=popemp.title()))
        ax.set_ylabel(yaxis_labels[nm].format(popemp=popemp.title()))
        ax.legend()

        if i == 2:
            label = "Year Quarters" if quarterly else "Year"
            ax.set_xlabel(label, labelpad=15)

    return fig


def _regions_plot_internal(
    excel_path: pathlib.Path,
    popemp: str,
    method: str,
    region_name: str,
    regions: list[str],
    quarterly: bool,
    plot_titles: dict[str, str],
    yaxis_labels: dict[str, str],
    npier_scenario: str = "NPIER",
    region_replace: Optional[dict[str, str]] = None,
):
    """Write regions totals to Excel and create line plots."""
    def column_year(column: str) -> int:
        match = re.match(r"(\d{4})(_q\d)?", column, re.I)
        if match is None:
            raise ValueError(f"unexpected column name: {column}")

        return int(match.group(1))

    if quarterly:
        perc_sheet = f"{npier_scenario} div EDDIE (%)"
    else:
        perc_sheet = f"% {npier_scenario} - EDDIE"

    data = pd.read_excel(
        excel_path, sheet_name=[npier_scenario, "EDDIE", perc_sheet], index_col=0
    )

    for nm in ("EDDIE", perc_sheet):
        data[nm] = data[nm].reindex(columns=data[npier_scenario].columns)

    kwargs = dict(
        popemp=popemp,
        regions=regions,
        quarterly=quarterly,
        plot_titles=plot_titles,
        yaxis_labels=yaxis_labels,
        npier_scenario=npier_scenario,
        region_replace=region_replace,
    )
    fig = _plot_yearly_comparisons(
        data, f"{method} - {popemp.title()} in {region_name} Regions", **kwargs
    )

    growth_data: dict[str, pd.DataFrame] = {}
    for nm in (npier_scenario, "EDDIE"):
        df = data[nm].copy()

        before_base = [i for i in df.columns if column_year(i) < BASE_YEAR]
        df = df.drop(columns=before_base)

        base_column = f"{BASE_YEAR}_q1" if quarterly else str(BASE_YEAR)
        growth = df.divide(df[base_column], axis=0) - 1

        growth_data[f"{nm} Growth"] = growth

    growth_data["NPIER - EDDIE Growth"] = (
        growth_data[f"{npier_scenario} Growth"] - growth_data["EDDIE Growth"]
    )

    growth_fig = _plot_yearly_comparisons(
        growth_data, f"{method} - {popemp.title()} Growth in {region_name} Regions", **kwargs
    )

    out_file = excel_path.with_name(excel_path.stem + f"-{region_name}_plots.pdf")
    with backend_pdf.PdfPages(out_file) as pdf:
        pdf.savefig(fig)
        pdf.savefig(growth_fig)

    plt.close("all")
    LOG.info(f"Written: {out_file}")


def ddg_regions_plots(base_folder: pathlib.Path):
    """Produce regions total summary plots for all DDG sub-folders."""
    sub_folder = "{method}\TfN Edited Comparison\DDG Comparison"
    methods = (
        ("Nov 21 NPIER - North values only", "North", NORTH_REGIONS),
        ("Nov 21 NPIER - North values only", "External", OTHER_GB_REGIONS),
        ("Nov 21 NPIER - North factored", "North", NORTH_REGIONS),
        ("Nov 21 NPIER - North factored", "External", OTHER_GB_REGIONS),
        ("Nov 21 NPIER - North values external factors", "North", NORTH_REGIONS),
        ("Nov 21 NPIER - North values external factors", "External", OTHER_GB_REGIONS),
        ("Nov 21 NPIER - North values only (NPIER Constraint)", "North", NORTH_REGIONS),
        ("Nov 21 NPIER - North values only (NPIER Constraint)", "External", OTHER_GB_REGIONS),
    )
    titles = {
        "NPIER": "NPIER Total {popemp} DDG by Year",
        "EDDIE": "EDDIE Total {popemp} DDG by Year",
        **dict.fromkeys(
            ("NPIER div EDDIE (%)", "% NPIER - EDDIE"),
            "NPIER vs EDDIE Percentage Difference for {popemp} by Year",
        ),
        "NPIER Growth": "NPIER DDG {popemp} Growth by Year",
        "EDDIE Growth": "EDDIE DDG {popemp} Growth by Year",
        "NPIER - EDDIE Growth": "NPIER - EDDIE {popemp} Growth by Year",
    }
    yaxis_labels = {
        **dict.fromkeys(("NPIER", "EDDIE"), "Total {popemp}"),
        **dict.fromkeys(
            ("NPIER div EDDIE (%)", "% NPIER - EDDIE"), "{popemp} Percentage Difference"
        ),
        **dict.fromkeys(("NPIER Growth", "EDDIE Growth"), "{popemp} Growth"),
        "NPIER - EDDIE Growth": "{popemp} Growth Difference",
    }

    for method, region_name, regions in methods:
        for popemp in ("Employment", "Population"):
            LOG.info(f"Plotting {method} - {region_name} {popemp.title()}")

            excel_file = (
                base_folder
                / sub_folder.format(method=method)
                / f"DD_Nov21_NPIER_Central_{popemp[:3]}_comparison-regions.xlsx"
            )
            _regions_plot_internal(
                excel_file,
                popemp,
                method,
                region_name,
                regions,
                quarterly=False,
                plot_titles=titles,
                yaxis_labels=yaxis_labels,
                region_replace={
                    "East": "east of england",
                    "East Mids": "east midlands",
                    "London": "london",
                    "North East": "north east",
                    "North West": "north west",
                    "Scotland": "scotland",
                    "South East": "south east",
                    "South West": "south west",
                    "Wales": "wales",
                    "West Mids": "west midlands",
                    "Yorks & Hum": "yorkshire & the humber",
                },
            )


def eddie_regions_plots(base_folder: pathlib.Path):
    """Plot region total summaries for each EDDIE input method."""
    LOG.info("Plotting for %s", base_folder.stem)
    titles = {
        "NPIER": "NPIER Total {popemp} by Year Quarters",
        "EDDIE": "EDDIE Total {popemp} by Year Quarters",
        "NPIER div EDDIE (%)": "NPIER vs EDDIE Percentage Difference for {popemp} by Year Quarters",
        "NPIER Growth": "NPIER DDG {popemp} Growth by Year Quarters",
        "EDDIE Growth": "EDDIE DDG {popemp} Growth by Year Quarters",
        "NPIER - EDDIE Growth": "NPIER - EDDIE {popemp} Growth by Year Quarters",
    }
    yaxis = {
        **dict.fromkeys(("NPIER", "EDDIE"), "Total {popemp}"),
        "NPIER div EDDIE (%)": "{popemp} Percentage Difference",
        **dict.fromkeys(("NPIER Growth", "EDDIE Growth"), "{popemp} Growth"),
        "NPIER - EDDIE Growth": "{popemp} Growth Difference",
    }
    plot_methods = [
        ("NPIER North Only Comparison", "North", NORTH_REGIONS),
        ("NPIER Factored North Comparison", "North", NORTH_REGIONS),
        ("NPIER North with Factored External Comparison", "North", NORTH_REGIONS),
        ("NPIER North with Factored External Comparison", "External", OTHER_GB_REGIONS),
    ]

    for folder, region_name, current_regions in plot_methods:
        folder: pathlib.Path = base_folder / folder

        for popemp in ("employment", "population"):
            LOG.info(f"Plotting {folder.stem} - {region_name} {popemp.title()}")

            excel_path = folder / f"NPIER_EDDIE_inputs_comparison-{popemp}_regions.xlsx"
            _regions_plot_internal(
                excel_path,
                popemp,
                method=folder.stem,
                region_name=region_name,
                regions=current_regions,
                quarterly=True,
                plot_titles=titles,
                yaxis_labels=yaxis,
            )


def eddie_dlog_regions_plots():
    """Plot """
    base_folder = pathlib.Path(
        r"C:\Users\ukmjb018\OneDrive - WSP O365\WSP_Projects\TfN NorMITs Demand Partner 2022\DLog Matrix\Outputs\DLogTEM\EDDIE Adjustments"
    )

    LOG.info(f"Plotting for {base_folder.stem}")
    TITLES = {
        "Dlog": "D-Log Total {popemp} by Year Quarters",
        "EDDIE": "EDDIE Total {popemp} by Year Quarters",
        "Dlog div EDDIE (%)": "D-Log vs EDDIE Percentage Difference for {popemp} by Year Quarters",
    }
    YAXIS = {
        **dict.fromkeys(("Dlog", "EDDIE"), "Total {popemp}"),
        "Dlog div EDDIE (%)": "{popemp} Percentage Difference",
    }
    plot_methods = [
        ("Nov 21 - DLog", "North", NORTH_REGIONS),
        ("Nov 21 - DLog", "External", OTHER_GB_REGIONS),
    ]

    for folder, region_name, current_regions in plot_methods:
        folder = base_folder / folder

        for popemp in ("employment", "population"):
            LOG.info(f"Plotting {folder.stem} - {region_name} {popemp.title()}")

            excel_path = folder / f"DLog_EDDIE_inputs_comparison-{popemp}_regions.xlsx"
            _regions_plot_internal(
                excel_path,
                popemp,
                method=folder.stem,
                region_name=region_name,
                regions=current_regions,
                quarterly=True,
                plot_titles=TITLES,
                yaxis_labels=YAXIS,
                npier_scenario="Dlog",
            )


def dlog_ddg_regions_plots():
    base_folder = r"C:\Users\ukmjb018\OneDrive - WSP O365\WSP_Projects\TfN NorMITs Demand Partner 2022\DLog Matrix\Outputs\DLogTEM\EDDIE Adjustments\{method}\TfN Edited Comparison\DDG Comparison"

    METHODS = (
        ("Nov 21 - DLog", "North", NORTH_REGIONS),
        ("Nov 21 - DLog", "External", OTHER_GB_REGIONS),
        ("Nov 21 - DLog", "Great Britain", NORTH_REGIONS + OTHER_GB_REGIONS),
    )
    titles = {
        "Dlog": "D-Log Total {popemp} DDG by Year",
        "EDDIE": "EDDIE Total {popemp} DDG by Year",
        **dict.fromkeys(
            ("Dlog div EDDIE (%)", "% Dlog - EDDIE"),
            "D-Log vs EDDIE Percentage Difference for {popemp} by Year",
        ),
    }
    yaxis_labels = {
        **dict.fromkeys(("Dlog", "EDDIE"), "Total {popemp}"),
        **dict.fromkeys(
            ("Dlog div EDDIE (%)", "% Dlog - EDDIE"), "{popemp} Percentage Difference"
        ),
    }

    for method, region_name, regions in METHODS:
        for popemp in ("Employment", "Population"):
            LOG.info(f"Plotting {method} - {region_name} {popemp.title()}")

            excel_file = (
                pathlib.Path(base_folder.format(method=method))
                / f"DD_Nov21_Dlog_Central_{popemp[:3]}_comparison-regions.xlsx"
            )
            _regions_plot_internal(
                excel_file,
                popemp,
                method,
                region_name,
                regions,
                quarterly=False,
                plot_titles=titles,
                yaxis_labels=yaxis_labels,
                region_replace={
                    "East": "east of england",
                    "East Mids": "east midlands",
                    "London": "london",
                    "North East": "north east",
                    "North West": "north west",
                    "Scotland": "scotland",
                    "South East": "south east",
                    "South West": "south west",
                    "Wales": "wales",
                    "West Mids": "west midlands",
                    "Yorks & Hum": "yorkshire & the humber",
                },
                npier_scenario="Dlog",
            )


def _run() -> None:
    logging.basicConfig(level=logging.INFO)
    eddie_folder = pathlib.Path(
        r"C:\Users\ukmjb018\OneDrive - WSP O365\WSP_Projects\TfN Secondment\NorMITs-Demand\Outputs\EDDIE\EDDIE - NPIER Raw Comparison\EDDIE - NPIER Raw Comparison - 20230303"
    )
    eddie_regions_plots(eddie_folder)
    ddg_folder = pathlib.Path(r"I:\NorMITs Demand\import\edge_replicant\edge_inputs\Drivers")
    ddg_regions_plots(ddg_folder)


if __name__ == "__main__":
    _run()
