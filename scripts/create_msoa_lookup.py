from typing import Tuple

import pandas as pd
import geopandas as gpd

# This script is used to create a lookup for development sites in the 
# development log. Although can be used for other spatial joins of 
# points to a polygon shapefile 
# It currently requires the geopandas package
# TODO Check if this script is necessary after rge d-log update

def create_msoa_lookup_from_bng(points_file: str,
                                msoa_shapefile: str,
                                msoa_id: str = "msoa11cd",
                                points_id: str = "development_site_id"
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print("Loading MSOA and point data")

    msoa_df = gpd.read_file(msoa_shapefile).to_crs(epsg=27700)
    msoa_df = msoa_df[[msoa_id, "geometry"]]

    points_cols = [points_id, "eastings", "northings"]
    points_df = pd.read_csv(points_file)[points_cols]

    # Remove any errors - no coordinates supplied
    invalid_points = points_df.loc[
        (points_df["eastings"].isna()) | (points_df["northings"].isna())
    ]
    points_df = points_df.dropna()

    print(f"missing Coords for: {invalid_points.shape[0]}")

    points_df = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df["eastings"],
                                    points_df["northings"]),
        crs={"init": "epsg:27700"}
    ).to_crs(epsg=27700)

    print("Performing Spatial Join")

    points_df = gpd.sjoin(points_df, msoa_df, how="left")

    # Extract points with no match
    no_match = points_df.loc[points_df[msoa_id].isna()]
    invalid_points = pd.concat(
        (invalid_points, no_match),
        axis=0,
        sort=True
    )
    points_df = points_df.dropna()

    print(f"Successful: {points_df.shape[0]}")
    print(f"No Match for: {no_match.shape[0]}")

    points_cols.append(msoa_id)

    return points_df[points_cols], invalid_points[points_cols]


if __name__ == "__main__":
    dlog_files = [
         r"C:\Users\Monopoly\Documents\EFS\data\dlog\dlog_residential.csv",
          r"C:\Users\Monopoly\Documents\EFS\data\dlog\dlog_nonresidential.csv",
    ]
    msoa_lookup = r"Y:\NorMITs Demand\import\shapes\uk_ew_msoa_s_iz.shp"
    lookup = pd.DataFrame()
    errors = pd.DataFrame()
    for dlog_file in dlog_files:
        df, errs = create_msoa_lookup_from_bng(
            dlog_file,
            msoa_lookup
        )
        lookup = pd.concat(
            [lookup, df], axis=0, sort=True
        )
        errors = pd.concat(
            [errors, errs], axis=0, sort=True
        )

    lookup.to_csv("development_msoa_lookup.csv", index=False)
    errors.to_csv("development_msoa_lookup_errors.csv", index=False)
