from ast import IsNot
from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
import os




@dataclass
class constants:
    folder = r"C:\Projects\MidMITS"
    LAD = gpd.read_file(os.path.join(folder,r'GIS\Midlands_LAD.shp'))
    dash_counts = pd.read_csv(os.path.join(folder,r'counts\Existing counts.csv'))
    districts = LAD['LAD21NM'].unique()
    drop_cols = ['Name', 'Description', 'Longitude', 'Latitude', 'Status', 'X',
    'Y', 'NAME_2', 'AREA_CODE', 'DESCRIPTIO', 'FILE_NAME', 'NUMBER',
    'NUMBER0', 'POLYGON_ID', 'UNIT_ID', 'CODE', 'HECTARES', 'AREA',
    'TYPE_CODE', 'TYPE_COD0', 'DESCRIPT1', 'DESCRIPT0', 'index_right', 'AM_SSe',
    'IP_SSe', 'PM_SSe']
    data_cols = ['AM_Car', 'AM_LGV', 'AM_HGV', 'AM_All', 'AM_SSe', 'IP_Car', 
    'IP_LGV', 'IP_HGV', 'IP_All','IP_SSe', 'PM_Car', 'PM_LGV', 'PM_HGV', 'PM_All',
    'PM_SSe', '12H_Car', '12H_LGV', '12H_HGV', '12H_All', '12H_SSe']
    growth_cols = [s + "_growth" for s in data_cols]
    dash_cols = {'AM car':'AM_Car_growth', 'AM LGV':'AM_LGV_growth', 'AM HGV':'AM_HGV_growth',
    'IP car':'IP_Car_growth', 'IP LGV':'IP_LGV_growth', 'IP HGV':'IP_HGV_growth', 'PM car':'PM_Car_growth',
    'PM LGV':'PM_LGV_growth','PM HGV':'PM_HGV_growth'}


def csv_as_gdf(path,name,crs):
    df = pd.read_csv(os.path.join(path,name))
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X,df.Y))
    gdf.crs = crs
    return gdf


def GrowthFactors(month):
    """
    Need to read in counts data for 2018 and 2021, from wherever possible - webTRIS, dashboard, DfT etc.
    Seperate data by LAD the counts are located within, and calculate growth factors between years for all
    Check how correlated the growths are by LAD; is the growth between years consistent over one LAD, or is there a large spread?
    Depending on the above, either groupby LAD and mean over to get a growth factor for each LAD, or take a closer look
    """
    c = constants
    gdf_18 = gpd.sjoin(csv_as_gdf(c.folder,f"counts\webtris\WebTRIS_Out_2018_{month}.csv",27700)
    ,c.LAD[['LAD21NM','geometry']]).drop(c.drop_cols,axis=1).set_index(['Id','LAD21NM'])
    gdf_21 = gpd.sjoin(csv_as_gdf(c.folder,f"counts\webtris\WebTRIS_Out_2021_{month}.csv",27700)
    ,c.LAD[['LAD21NM','geometry']]).drop(c.drop_cols,axis=1).set_index(['Id','LAD21NM'])
    df = pd.concat([gdf_18,gdf_21], keys = ['18','21'],axis=1,join='inner').stack(1)
    writer = pd.ExcelWriter(os.path.join(c.folder,f"growth_stats_{month}_2.xlsx"), engine="openpyxl",mode="w")
    writer_full = pd.ExcelWriter(os.path.join(c.folder,f"WebTRIS_summary_{month}.xlsx"), engine="openpyxl",mode="w")
    no_data = []
    for district in c.districts:
        district_df = df[df['LAD21NM']==district]
        for column in c.data_cols:
            district_df[f"{column}_growth"] = (district_df[f"{column}_21"] - district_df[f"{column}_18"])/district_df[f"{column}_21"]
        stats = district_df[c.growth_cols].describe()
        if stats.isnull().iloc[4,4]==False:
            stats.to_excel(writer,sheet_name=f"{district}")
            district_df.set_index('Id').drop(['geometry_18','geometry_21','LAD21NM',],axis=1).to_excel(writer_full,sheet_name=f"{district}")
        else:
            print(f"{district} has no WebTRIS data")
            no_data.append(district)
    #missing_LAD = c.LAD[c.LAD['LAD21NM'].isin(no_data)]
    #missing_LAD.to_file(os.path.join(c.folder,f"missing_data{month}.shp"))
    writer.close()
    writer_full.close()


if __name__=="__main__":
    for month in [3,11]:
        GrowthFactors(month)

#we want [ID,X,Y,AM_Car etc.]
