###Imports####
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import dataclasses
##############

@dataclass
class constants:
    """Class for keeping contants used later in the file
    time_update: Dictionary to translate from day of week to 'Weekday' or 'Weekends'
    path: the common file path and start of file name of all the files being read in
    output_path: path to folder CSVs will be written to
    dates: Dict of months, and ends of file names where the month's data are saved - the data for March 2019 are saved in files ending with 20190301-20190331 etc.
    zone_update: Correspondence table of MSOA to MND zones
    """
    time_update = {**dict.fromkeys((0,1,2,3,4,5,6,19,20,21,22,23),'OP'),
    **dict.fromkeys((7,8,9),'AM'),**dict.fromkeys((10,11,12,13,14,15),'IP'),
    **dict.fromkeys((16,17,18),'PM')}
    week_update = {**dict.fromkeys((0,1,2,3,4),'Weekday'),**dict.fromkeys((5,6), 'Weekend')}
    path = r"C:\Projects\MidMITS\O2 MPD Trip End Data\O2 Trip End Data\Daily Journeys MSOA with Purpose and TOD on "
    output_path = r"C:\Projects\MidMITS\O2 MPD Trip End Data\O2 Trip End Data\MND_"
    dates = {'March_2019':list(range(20190301,20190332)),'Nov_2019':list(range(20191101,20191131)),'March_2020':list(range(20200301,20200332)),
    'Nov_2020':list(range(20201101,20201131)),'March_2021':list(range(20210301,20210332)),'Nov_2021':list(range(20211101,20211113))+list(range(20211115,20211131))}
    zone_update = pd.read_csv(r"C:\Projects\MidMITS\Python\uk_msoa_2_mnd.csv")[['Msoa11cd','MC_MND']]


def main():
    """
    Loops through input CSVs, getting each into the correct format then adding to a final dataframe, before exporting to a CSV at the end of the loop
    for each month.
    """
    c = constants
    c.zone_update.columns = ['msoa','MND']
    for month in c.dates.keys():
        final = pd.DataFrame(columns = ['date', 'msoa', 'hour_part', 'journey_purpose', 'journeys_starting', 'journeys_ending', 'day'])
        for date in c.dates[month]:
            df = pd.read_csv(f"{c.path}{date}.csv")
            #removing entries stochastically rounded up to 10 for privacy reasons. Currently commented out as it had a minimal effect on data
            #df.loc[df['journeys_starting']==10,'journeys_starting'] = 0
            #df.loc[df['journeys_ending']==10,'journeys_ending'] = 0
            df.replace({'hour_part':c.time_update},inplace=True)
            df.drop('msoa_name',axis=1,inplace=True)
            df = df.groupby(['date','msoa','hour_part','journey_purpose']).sum().reset_index()
            df['day'] = df['date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").date().weekday())
            df.replace({'day':c.week_update},inplace=True)
            df.columns = ['date', 'msoa', 'hour_part', 'journey_purpose', 'journeys_starting', 'journeys_ending', 'day']
            final = pd.concat([final,df],axis=0)
        final.drop('date',axis=1,inplace=True)
        final = final.groupby(['day','msoa','hour_part','journey_purpose']).mean().reset_index()
        final = final.merge(c.zone_update,on='msoa')
        final.drop('msoa',axis=1,inplace=True)
        final.groupby(['day','MND','hour_part','journey_purpose']).sum().reset_index().to_csv(f"{c.output_path}{month}_summed.csv")


if __name__ == "__main__":
    main()