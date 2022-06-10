import NTEM_process as nt
import pandas as pd
import pyodbc
from os import path

for year in nt.params.years.keys():
    if nt.params.years[year] is not None and str(nt.params.years[year]) in nt.NTEMFILES.keys():
        df = nt.read_NTEM(nt.params.years[year])
        df.to_csv(path.join(nt.params.data_source,nt.params.NTEM_output_dir,f"{nt.params.years[year]}.csv"))
