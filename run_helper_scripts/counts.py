from ast import IsNot
from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
import os




@dataclass
class constants:
    folder = r"C:\Projects\MidMITS"
    LAD = gpd.read_file(os.path.join(folder,r'GIS\LAD\Midlands_LAD.shp'))
    county = gpd.read_file(r"C:\Users\ukiws001\Desktop\OS boundaries\Data\Supplementary_Ceremonial\Boundary-line-ceremonial-counties_region.shp")
    dash_counts = pd.read_csv(os.path.join(folder,r'counts\Existing counts.csv'))
    #segmentations = {"county":county['NAME'].unique(),"LAD":LAD['LAD21NM'].unique()}
    counties = {"shapefile":county,"values":county['NAME'].unique(),"file name":"County"}
    districts = {"shapefile":LAD,"values":LAD['NAME'].unique(),"file name":"LAD"}
    
    drop_cols = ['X','Y','AM_SSe','IP_SSe', 'PM_SSe', '12H_SSe', 'geometry','index_right']
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


def GrowthFactors(month,seg_level):
    """
    Need to read in counts data for 2018 and 2021, from wherever possible - webTRIS, dashboard, DfT etc.
    Seperate data by LAD the counts are located within, and calculate growth factors between years for all
    Check how correlated the growths are by LAD; is the growth between years consistent over one LAD, or is there a large spread?
    Depending on the above, either groupby LAD and mean over to get a growth factor for each LAD, or take a closer look
    """
    c = constants
    period = ['AM','IP','PM','12H']
    veh_type = ['Car','LGV','HGV','All']
    combined = pd.MultiIndex.from_product([period,veh_type], names=['Time Period', 'Vehicle Type'])
    gdf_18 = gpd.sjoin(csv_as_gdf(c.folder,f"counts\webtris\WebTRIS_Out_2018_{month}_NOM1.csv",27700)
    ,seg_level["shapefile"],how='left',op='within').drop(c.drop_cols,axis=1).set_index(['Id','NAME']).drop_duplicates()
    gdf_21 = gpd.sjoin(csv_as_gdf(c.folder,f"counts\webtris\WebTRIS_Out_2021_{month}_NOM1.csv",27700)
    ,seg_level["shapefile"],how='left',op='within').drop(c.drop_cols,axis=1).set_index(['Id','NAME']).drop_duplicates()
    gdf_18.columns = combined
    gdf_21.columns = combined
    df = pd.concat([gdf_18,gdf_21], keys = ['18','21'],axis=1,join='inner')
    diff = df.stack([1,2]).groupby(['NAME','Time Period','Vehicle Type']).sum()
    diff['Growth'] = (diff['21'] - diff['18']) / diff['18']
    diff.to_csv(r"C:\Projects\MidMITS\counts\stacked_district"+str(month)+"NOM1.csv") #Outputs to stacked table
    writer = pd.ExcelWriter(os.path.join(c.folder,f"growth_stats_{month}{seg_level['file name']}_NOM1.xlsx"), engine="openpyxl",mode="w")
    writer_full = pd.ExcelWriter(os.path.join(c.folder,f"WebTRIS_summary_{month}{seg_level['file name']}_NOM1.xlsx"), engine="openpyxl",mode="w")
    no_data = []
    for seg in seg_level["values"]:
        if seg in df.index.get_level_values('NAME'):
            seg_df = df.loc[:,seg,:]
            for period in ['AM','IP','PM', '12H']:
                for veh in ['All','Car','LGV','HGV']:
                    seg_df['Growth',period,veh] = (seg_df['21'][period][veh] - seg_df['18'][period][veh])/seg_df['18'][period][veh]
            stats = seg_df['Growth'].describe()
            stats.to_excel(writer,sheet_name=f"{seg}")
            seg_df.to_excel(writer_full,sheet_name=f"{seg}")
        else:
            print(f"{seg} isn't there")
            no_data.append(seg)
    missing_LAD = seg_level["shapefile"][seg_level["shapefile"]['NAME'].isin(no_data)]
    missing_LAD.to_file(os.path.join(c.folder,f"missing_data{month}{seg_level['file name']}.shp"))
    writer.close()
    writer_full.close()


if __name__=="__main__":
    for month in [11]:
        GrowthFactors(month,constants.counties)

#we want [ID,X,Y,AM_Car etc.]
