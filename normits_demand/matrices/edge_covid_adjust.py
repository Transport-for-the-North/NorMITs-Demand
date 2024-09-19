import pandas as pd
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter
from pathlib import Path

lookup = {'HBEBCA_Int': 'business','HBEBNCA_Int': 'business','NHBEBCA_Int': 'business','NHBEBNCA_Int': 'business','HBWCA_Int': 'commute',
                  'HBWNCA_Int': 'commute', 'HBOCA_Int': 'other', 'HBONCA_Int': 'other', 'NHBOCA_Int': 'other', 'NHBONCA_Int': 'other',
                  'EBCA_Ext_FM': 'business',
                  'EBCA_Ext_TO': 'business',
                  'EBNCA_Ext': 'business',
                  'HBWCA_Ext_FM': 'commute',
                  'HBWCA_Ext_TO': 'commute',
                  'HBWNCA_Ext': 'commute',
                  'OCA_Ext_FM': 'other',
                  'OCA_Ext_TO': 'other',
                  'ONCA_Ext': 'other'
                  }

factors = {'commute': 0.7811, 'business': 0.73, 'other': 1.0}



def factor_dir(mat_dir, years):
    con = CUBEMatConverter(Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"))
    for year in years:
        factored = {}
        for segment, uc in lookup.items():
            mat = pd.read_csv(mat_dir / f"{year}_24Hr_{segment}.csv", index_col=[0,1])
            mat *= factors[uc]
            if year == 2045: # check if this should be 44
                mat *= 1.016472257
                year = 2053 # check if this should be 52
            mat.to_csv(mat_dir / f"{segment}.csv", header=False)
            factored[segment] = mat_dir / f"{segment}.csv"
        con.csv_to_mat(1300, factored, mat_dir / f"PT_24hr_Demand_{year}.MAT")

if __name__ == "__main__":
    factor_dir(Path(r"E:\NorMITs Demand\Rotherham\Forecasting\edge\1.0\iter1.review\High\train"),
               [2028])

