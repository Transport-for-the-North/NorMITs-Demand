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
    msoa2ladpath = pathlib.Path(r"C:\Projects\MidMITS\Python\outputs\msoa2lad.csv")
    dft_url = r"https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1094056/COVID-19-transport-use-statistics.ods"
    dft_date = (11,2021)


def duck(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Don't remember, adding docstring to get rid of error.  Will update later
    Args:
        df (pd.DataFrame): Dataframe
        day (str): Day name

    Returns:
        pd.DataFrame: Dataframe
    """
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

def dft_factors(date:tuple = constants.dft_date) -> None:
    """
    Function to create DfT factors csv in the form required by Apply_mnd.py
    Args:
        date (Tuple, optional): The month and year you want the DfT factor for. Defaults to constants.dft_date.
    """
    columns = ['Car','Rail','Bus_excl_London','Cycling']
    df = pd.read_excel(constants.dft_url,engine='odf',skiprows=6)
    df.columns=['Date','Car','LGV','HGV','All Veh','Rail','TfL Tube','TfL Bus','Bus_excl_London','Cycling','misc.']
    df = df[(df.Cycling != '..') & (df.Bus_excl_London != '..')]
    df['Cycling'] = pd.to_numeric(df['Cycling'])
    df['Bus_excl_London'] = pd.to_numeric(df['Bus_excl_London'])
    df['Day'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.replace({'Day':constants.week_update},inplace=True)
    needed = df.groupby(['Year','Month','Day']).mean().loc[date[1],date[0]]
    factors = needed[columns] / df.iloc[0][columns]
    factors['Walking'] = 1
    factors.rename(columns={'Walking':1, 'Cycling':2, 'Car':3, 'Bus_excl_London':5,'Rail':6,},inplace=True)
    factors = factors.unstack().reset_index()
    factors.columns=['m','tp','factor']
    factors.set_index('tp',inplace=True)
    tp_update = ['Weekday'] * 4
    tp_update.append('Saturday')
    tp_update.append('Sunday')
    tp = pd.DataFrame(data=tp_update,index=range(1,7))
    tp.reset_index(inplace=True)
    tp.columns = ['After','Before']
    tp.set_index('Before',inplace=True)
    new_factors = factors.join(tp)
    new_factors.set_index('After',inplace=True)
    test = new_factors.copy()
    test['msoa'] = 'dummy'
    msoa = pd.read_csv(constants.msoa2ladpath)
    for i in msoa['zone_id']:
        df = new_factors
        df['msoa'] = i
        test = pd.concat([test,df])
    test.reset_index(inplace=True)
    test.set_index('MSOA',inplace=True)
    test.drop('dummy',inplace=True)
    test.rename(columns={'After':'tp'},inplace=True)
    test.to_csv(constants.output_path / f'dft_factors{date[0]}/{date[1]}.csv')

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
    update = ['Other'] * 7
    update.insert(0,'Commute')
    join = pd.DataFrame(data=update,index=range(1,9),columns=['p'])
    join.reset_index(inplace=True)
    join.set_index('p',inplace=True)
    zone_join=pd.read_csv(c.msoa2ladpath)
    zone_join.rename(columns={'zone_id':'msoa'}, inplace=True)
    zone_join.set_index('msoa',inplace=True)
    #total.set_index('msoa',inplace=True)
    total = total.join(zone_join)
    total.reset_index(inplace=True)
    total.rename(columns={'journey_purpose':'p'},inplace=True)
    total.set_index('p',inplace=True)
    output = total.join(join)
    output.reset_index(inplace=True)
    output.drop(['p','LAD','LAD20NM'],axis=1,inplace=True)
    output.rename(columns={'hour_part':'tp','index':'p','LAD20CD':'LAD'},inplace=True)
    output.set_index(['msoa','County','tp','p','LAD'],inplace=True)
    output.to_csv(os.path.join(c.output_path, "complete.csv"))
    dft_factors(11,2021)


if __name__ == "__main__":
    main()
