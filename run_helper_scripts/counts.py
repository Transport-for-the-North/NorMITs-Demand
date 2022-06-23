from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
import os

"""
Process to take in WebTRIS counts and output data to factor counts up over years.
It is currently set up specifically for 2018 to 2021, but will work exactly the same provided data for any two
years.
"""


@dataclass
class Constants:
    """
    dataclass for constants and variables.
    years: Years of the counts scaling from and to
    folder: Base directory
    county: UK counties shapefile
    LAD: Local authority districts shapefile
    dash_counts: The existing counts to be scaled up using this process
    """

    years = {"base year": 2015, "target_year": 2018}
    folder = r"C:\Projects\MidMITS"
    LAD = gpd.read_file(os.path.join(folder, r"GIS\LAD\Midlands_LAD.shp"))
    county = gpd.read_file(
        r"C:\Users\ukiws001\Desktop\OS boundaries\Data\Supplementary_Ceremonial\Boundary-line-ceremonial-counties_region.shp"
    )
    dash_counts = pd.read_csv(os.path.join(folder, r"counts\Existing counts_updated.csv"))
    # segmentations = {"county":county['NAME'].unique(),"LAD":LAD['LAD21NM'].unique()}
    counties = {"shapefile": county, "values": county["NAME"].unique(), "file name": "County"}
    districts = {"shapefile": LAD, "values": LAD["NAME"].unique(), "file name": "LAD"}

    drop_cols = ["X", "Y", "AM_SSe", "IP_SSe", "PM_SSe", "12H_SSe", "geometry", "index_right"]
    data_cols = [
        "AM_Car",
        "AM_LGV",
        "AM_HGV",
        "AM_All",
        "AM_SSe",
        "IP_Car",
        "IP_LGV",
        "IP_HGV",
        "IP_All",
        "IP_SSe",
        "PM_Car",
        "PM_LGV",
        "PM_HGV",
        "PM_All",
        "PM_SSe",
        "12H_Car",
        "12H_LGV",
        "12H_HGV",
        "12H_All",
        "12H_SSe",
    ]
    growth_cols = [s + "_growth" for s in data_cols]
    dash_cols = {
        "AM car": "AM_Car_growth",
        "AM LGV": "AM_LGV_growth",
        "AM HGV": "AM_HGV_growth",
        "IP car": "IP_Car_growth",
        "IP LGV": "IP_LGV_growth",
        "IP HGV": "IP_HGV_growth",
        "PM car": "PM_Car_growth",
        "PM LGV": "PM_LGV_growth",
        "PM HGV": "PM_HGV_growth",
    }


def csv_as_gdf(path: str, name: str, crs: int) -> gpd.GeoDataFrame:
    """
    Function to read in a csv with coordinate columns and output a geodataframe
    Args:
        path (_type_): The directory the csv sits in
        name (_type_): The name of the csv within the 'path' folder.  This csv must have coordinate columns called
        'X' and 'Y'.  This can be made more flexible later
        crs (_type_): The crs the geodataframe should be in.
    Returns:
        GeoDataFrame: The input csv as a geopandas geodataframe
    """
    df = pd.read_csv(os.path.join(path, name))
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
    gdf.crs = crs
    return gdf


def growthfactors(
    month: int,
    zone_system: dict,
    base_year: int = Constants.years["base year"],
    target_year: int = Constants.years["target year"],
):
    """
    Need to read in counts data for 2018 and 2021, from wherever possible - webTRIS, dashboard, DfT etc.
    Seperate data by LAD the counts are located within, and calculate growth factors between years for all
    Check how correlated the growths are by LAD; is the growth between years consistent over one LAD, or is there a large spread?
    Depending on the above, either groupby LAD and mean over to get a growth factor for each LAD, or take a closer look
    Args:
        month (int): The month the counts are for
        zone_system (dict): The zoning you want it done at (this is currently county or district)
        base_year (int): The base year 
        target_year (int): The target year
    """
    c = Constants
    period = ["AM", "IP", "PM", "12H"]
    veh_type = ["Car", "LGV", "HGV", "All"]
    combined = pd.MultiIndex.from_product(
        [period, veh_type], names=["Time Period", "Vehicle Type"]
    )
    gdf_base = (
        gpd.sjoin(
            csv_as_gdf(c.folder, rf"counts\webtris\WebTRIS_Out_2015_{month}_NOM1.csv", 27700),
            zone_system["shapefile"],
            how="left",
            op="within",
        )
        .drop(c.drop_cols, axis=1)
        .set_index(["Id", "NAME"])
        .drop_duplicates()
    )
    gdf_target = (
        gpd.sjoin(
            csv_as_gdf(c.folder, rf"counts\webtris\WebTRIS_Out_2018_{month}_NOM1.csv", 27700),
            zone_system["shapefile"],
            how="left",
            op="within",
        )
        .drop(c.drop_cols, axis=1)
        .set_index(["Id", "NAME"])
        .drop_duplicates()
    )
    gdf_base.columns = combined
    gdf_target.columns = combined
    df = pd.concat(
        [gdf_base, gdf_target], keys=[str(base_year), str(target_year)], axis=1, join="inner"
    )
    diff = df.stack([1, 2]).groupby(["NAME", "Time Period", "Vehicle Type"]).sum()
    diff["Growth"] = (diff[str(target_year)] - diff[str(base_year)]) / diff[str(base_year)]
    diff.to_csv(
        r"C:\Projects\MidMITS\counts\stacked_district_15" + str(month) + "NOM1.csv"
    )  # Outputs to stacked table
    writer = pd.ExcelWriter(
        os.path.join(c.folder, f"growth_stats_{month}{zone_system['file name']}_NOM1.xlsx"),
        engine="openpyxl",
        mode="w",
    )
    writer_full = pd.ExcelWriter(
        os.path.join(
            c.folder,
            f"WebTRIS_summary_{base_year}_to_{target_year}_{month}_{zone_system['file name']}.xlsx",
        ),
        engine="openpyxl",
        mode="w",
    )
    no_data = []
    for seg in zone_system["values"]:
        if seg in df.index.get_level_values("NAME"):
            seg_df = df.loc[:, seg, :]
            for period in ["AM", "IP", "PM", "12H"]:
                for veh in ["All", "Car", "LGV", "HGV"]:
                    seg_df["Growth", period, veh] = (
                        seg_df["18"][period][veh] - seg_df["15"][period][veh]
                    ) / seg_df["15"][period][veh]
            stats = seg_df["Growth"].describe()
            stats.to_excel(writer, sheet_name=f"{seg}")
            seg_df.to_excel(writer_full, sheet_name=f"{seg}")
        else:
            print(f"{seg} isn't there")
            no_data.append(seg)
    missing_LAD = zone_system["shapefile"][zone_system["shapefile"]["NAME"].isin(no_data)]
    missing_LAD.to_file(
        os.path.join(c.folder, f"missing_data{month}{zone_system['file name']}.shp")
    )
    writer.close()
    writer_full.close()


if __name__ == "__main__":
    for menesis in [11]:
        growthfactors(menesis, Constants.counties)

# we want [ID,X,Y,AM_Car etc.]
