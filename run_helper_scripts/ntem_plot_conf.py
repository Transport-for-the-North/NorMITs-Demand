from caf.toolkit import BaseConfig
from pathlib import Path


class GeoSpatialFile(BaseConfig):
    """Path to a geospatial file and the relevant ID column name."""

    path: Path
    id_column: str


class plotting_params(BaseConfig):
    """
    Params for the heatmap plotting portion of ntem plotting module

    Parameters
    ----------

    geospatial_file (GeoSpatialFile): the geospatial data you want to use
    to plot. This module doesn't perform any sorts of translation so this
    must match zoning in the data being plotted
    plot_zoning (str): The name of the zoning level the summary data is at.
    This must match the name in the TEMPro comparisons folder produced
    by the forecasting module as that folder is searched for comparison
    csvs to plot.
    """

    geospatial_file: GeoSpatialFile
    plot_zoning: str


class PAPlotsParameters(BaseConfig):
    """
    Config class for running ntem_plots module.

    Parameters
    ----------

    base_matrix_folder (Path): Folder containing base matrices
    forecast_matrix_folder (Path): Folder containing forecast matrices
    matrix_zoning (str): The zoning the matrices are given at.
    tempro_comparison_folder (str): The folder containing Tempro comparisons.
        if the forecasting module has been run this should be called
        'TEMPro Comparisons'
    tempro_comp_summary_zoning (str): The zone system the tempro summaries
        are at. These will be excel files in the above folder
    analytical_area_shape (Path): Path to a shape the plots will be zoomed on.
    base_year (int): The base year for forecasting
    diff_plot_params (plotting_params): The zoning info for plotting diff
        plots. This must match the csv files in TEMPro comparisons
    total_plot_params (plotting_params): Zoning info for plotting of overall
        numbers.
    """

    output_folder: Path
    base_matrix_folder: Path
    forecast_matrix_folder: Path
    matrix_zoning: str
    tempro_comparison_folder: Path
    tempro_comp_summary_zoning: Path
    analytical_area_shape: GeoSpatialFile
    base_year: int
    diff_plot_params: plotting_params
    total_plot_params: plotting_params
