from openpyxl import load_workbook
from counts import constants
import pandas as pd


#df = pd.DataFrame(index=constants.growth_cols)
wb = load_workbook(r"C:\Projects\MidMITS\counts\growth_stats_11.xlsx",read_only=True)
growth = pd.read_csv(r"C:\Projects\MidMITS\counts\stacked11.csv").groupby(['NAME','Time Period', 'Vehicle Type']).sum()
#growth['Growth'] = (growth['21'] - growth['18']) / growth['18']
dash = pd.read_csv(r"C:\Projects\MidMITS\counts\Existing counts county.csv")


for county in dash['County'].dropna().unique():
    #if district in wb.sheetnames:
    #sheet = pd.read_excel(r"C:\Projects\MidMITS\growth_stats_11.xlsx",sheet_name=district).set_index('Unnamed: 0')
    #df[district] = pd.Series(sheet.loc['mean'])
    cols = constants.dash_cols
    for time in ['AM','IP','PM']:
        for type in ['Car', 'LGV', 'HGV']:
            #dash[dash['LAD21NM']==district][f"{column}_2021"] = dash[dash['LAD21NM']==district][f"{column}"] * (1+df.loc[cols[column],district])
            dash.loc[dash['County']==county,f"{time} {type} 2021"] = dash.loc[dash['County']==county,f"{time} {type}"] * (1+growth.loc[(county,time,type),'Growth'])
    #else:
         #print(f"no data for {district}")
dash.drop(['Unnamed: 0'],axis=1).set_index(['A','B']).to_csv(r"C:\Projects\MidMITS\counts\county_output.csv")
#df.to_csv(r"C:\Projects\MidMITS\counts\factors.csv")