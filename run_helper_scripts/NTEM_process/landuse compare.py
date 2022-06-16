
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os
sys.path.append('..')
import normits_demand as nd
import pathlib
path = pathlib.Path(r'C:\Projects\MidMITS\Python\outputs\ApplyMND\iter9.6c-COVID\NTEM')
lookup = lookup = pd.read_csv(r"C:\Projects\MidMITS\GIS\msoa_lad_lookup.csv").set_index('msoa_zone_id')
years = [21,30,40]
cmaps = {'r':'Reds','b':'Blues'}
types = ['pop','emp']
for i in ['hb','nhb']:
    for j in ['attractions','productions']:
        df = pd.read_csv(os.path.join(path,f'{i}_{j}_comp.csv'))#.set_index('msoa_zone_id').join(lookup)
        lad = df.groupby('lad_2020_zone_id').sum()
        for k in [1,2]:
            lad[f'val_comp_{k}'] = lad[f'val_{years[k]}'] / lad[f'val_{years[k-1]}']
            lad[f'pop_comp_{k}'] = lad[f'people_{years[k]}_pop'] / lad[f'people_{years[k-1]}_pop']
            lad[f'emp_comp_{k}'] = lad[f'people_{years[k]}_emp'] / lad[f'people_{years[k-1]}_emp']
            lad[f'emp_diff_{k}'] = lad[f'emp_comp_{k}'] - lad[f'val_comp_{k}']
            lad[f'pop_diff_{k}'] = lad[f'pop_comp_{k}'] - lad[f'val_comp_{k}']
            fig = plt.figure(figsize=(16,12))
            ax=fig.add_subplot(1, 1, 1)
            label_patches=[]
            for m,n in zip(cmaps.keys(),types):
                data = lad[f'{n}_diff_{k}']
                label = f'{n}_{k}'
                sns.kdeplot(data,color=m,ax=ax,fill=True)
                label_patch = mpatches.Patch(
                color=sns.color_palette(cmaps[m])[5],label=label)
                label_patches.append(label_patch)
            plt.title(f'{i}_{j}_{k}')
            plt.legend(handles=label_patches,loc='upper left',prop={'size':15})
            plt.savefig(os.path.join(path,'Plots',f'{i}_{j}_{k}'))
        lad[['val_comp_1','val_comp_2','pop_comp_1','pop_comp_2','emp_comp_1','emp_comp_2',
            'emp_diff_1','emp_diff_2','pop_diff_1','pop_diff_2']].describe().to_csv(os.path.join(path,'Plots',f'{i}_{j}_describe.csv'))
        lad.to_csv(os.path.join(path,f'{i}_{j}_lad.csv'))

