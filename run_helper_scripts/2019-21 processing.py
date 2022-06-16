###Imports####
from dataclasses import dataclass
from datetime import datetime
import os
import pandas as pd
import pathlib

##############
"""

"""
@dataclass
class constants:
    """Class for keeping contants used later in the file
    time_update: Dictionary to translate from day of week to 'Weekday' or 'Weekends'
    path: the common file path and start of file name of all the files being read in
    output_path: path to folder CSVs will be written to
    dates: Dict of months, and ends of file names where the month's data are saved - 
    the data for March 2019 are saved in files ending with 20190301-20190331 etc.
    zone_update: Correspondence table of MSOA to MND zones
    """

    OP = {n: "OP" for n in list(range(7)) + list(range(19, 24))}
    AM = {n: "AM" for n in range(7, 10)}
    IP = {n: "IP" for n in range(10, 16)}
    PM = {n: "PM" for n in range(16, 19)}
    time_update = {**OP, **AM, **IP, **PM}
    week_update = {**{n: "Weekday" for n in range(5)}, **{5: "Saturday", 6: "Sunday"}}
    path = pathlib.Path(r"C:\Projects\MidMITS\O2 MPD Trip End Data\O2 Trip End Data")
    output_path = pathlib.Path(r"C:\Projects\MidMITS\Python\outputs")
    dates = {
        "March_2019": list(range(20190301, 20190332)),
        "Nov_2019": list(range(20191101, 20191131)),
        "March_2020": list(range(20200301, 20200332)),
        "Nov_2020": list(range(20201101, 20201131)),
        "March_2021": list(range(20210301, 20210332)),
        "Nov_2021": list(range(20211101, 20211113)) + list(range(20211115, 20211131)),
    }
    zone_update = pd.read_csv(
        os.path.join(output_path, "aggregation.csv"),
        names=["msoa", "County", "LAD"],
        index_col="msoa",
    )


def duck(df: pd.DataFrame, day: str):
    sub = (
        df.loc[f"{day}", "AM"] * 3
        + df.loc[f"{day}", "PM"] * 3
        + df.loc[f"{day}", "IP"] * 6
        + df.loc[f"{day}", "OP"] * 12
    )
    sub["hour_part"] = day
    sub.reset_index(inplace=True)
    sub.set_index(["hour_part", "msoa", "journey_purpose"], inplace=True)
    return sub


def main():
    """
    Loops through input CSVs, getting each into the correct format then adding to a final dataframe,
     before exporting to a CSV at the end of the loop for each month.
    """
    c = constants
    for month in c.dates.keys():
        final = pd.DataFrame(
            columns=[
                "date",
                "msoa",
                "hour_part",
                "journey_purpose",
                "origin",
                "destination",
                "day",
            ]
        )
        for date in c.dates[month]:
            df = pd.read_csv(
                os.path.join(c.path, f"Daily Journeys MSOA with Purpose and TOD on {date}.csv")
            )
            df.replace({"hour_part": c.time_update}, inplace=True)
            df.drop("msoa_name", axis=1, inplace=True)
            df = (
                df.groupby(["date", "msoa", "hour_part", "journey_purpose"])
                .sum()
                .reset_index()
            )
            df["day"] = df["date"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d").date().weekday()
            )
            df.replace({"day": c.week_update}, inplace=True)
            df.columns = [
                "date",
                "msoa",
                "hour_part",
                "journey_purpose",
                "origin",
                "destination",
                "day",
            ]
            final = pd.concat([final, df], axis=0)
        final.drop("date", axis=1, inplace=True)
        export = final.groupby(["day", "hour_part", "msoa", "journey_purpose"]).mean()
        sat = duck(export, "Saturday")
        sun = duck(export, "Sunday")
        weekend = pd.concat([sat, sun], axis=0)
        export_2 = pd.concat([weekend / 24, export.loc["Weekday"]], axis=0).join(
            c.zone_update, how="inner"
        )
        export_2.to_csv(os.path.join(c.output_path, f"{month}.csv"))
    total = pd.read_csv(
        os.path.join(c.output_path, "March_2019.csv"),
        index_col=["msoa", "LAD", "County", "hour_part", "journey_purpose"],
    ).drop(["origin", "destination"], axis=1)
    for month in c.dates.keys():
        df = pd.read_csv(
            os.path.join(c.output_path, f"{month}.csv"),
            index_col=["msoa", "LAD", "County", "hour_part", "journey_purpose"],
        )
        total = total.join(df, how="inner", rsuffix=f" {month}")
    total.to_csv(os.path.join(c.output_path, "complete.csv"))


if __name__ == "__main__":
    main()
