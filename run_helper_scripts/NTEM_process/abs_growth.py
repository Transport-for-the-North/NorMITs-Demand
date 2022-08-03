import NTEM_process as nt
import os

for sector in ('pop','emp'):
    df = nt.apply_abs(sector).droplevel(f'{sector} code').drop(['2018','2018_grouped','prop','diff'],axis=1)
    df.to_csv(os.path.join(nt.Params.data_source,"SHP",sector,f"abs_{sector}_formatted.csv"))