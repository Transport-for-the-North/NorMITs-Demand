import pandas as pd
import geopandas as gpd


lookup = pd.read_csv(r"C:\Projects\MidMITS\counts\LADtoCOUNTY.csv", usecols = ['Local Authority','County'])
counts = pd.read_csv(r"C:\Projects\MidMITS\counts\Existing counts.csv").rename(columns = {'LAD21NM':'Local Authority'})

joined_counts = pd.merge(counts,lookup,how='left',on='Local Authority')

joined_counts.to_csv(r"C:\Projects\MidMITS\counts\Existing counts county.csv")




