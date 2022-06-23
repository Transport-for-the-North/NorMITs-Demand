import NTEM_process as nt
import pandas as pd
import pyodbc
from os import path

for year in nt.Params.years.keys():
    if nt.Params.years[year] is not None and str(nt.Params.years[year]) in nt.NTEMFILES.keys():
        
        df = nt.read_NTEM(nt.Params.years[year])
        df.to_csv(path.join(nt.Params.data_source,nt.Params.NTEM_output_dir,f"{nt.Params.years[year]}.csv"))
