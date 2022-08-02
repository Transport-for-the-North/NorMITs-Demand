import pandas as pd
import pathlib
import numpy as np
import tqdm
import normits_demand as nd
import csv

pa = pathlib.Path(r'T:\MidMITs Demand\Distribution Model\iter9.7-COVID.2\car_and_passenger\Final Outputs\Full PA Matrices')
od = pathlib.Path(r'T:\MidMITs Demand\Distribution Model\iter9.7-COVID.2\car_and_passenger\Final Outputs\Full OD Matrices')

for i in [pa,od]:
    checks = {}
    for path in i.glob('*.csv.bz2'):
        df = pd.read_csv(path,index_col=0)
        nan = df.isnull().values.sum()
        neg = (df < 0).sum().sum()
        zero = (df == 0).sum().sum()
        total = df.sum().sum()
        checks[str(path).split("\\")[-1].split(".")[0]] = np.array([nan,neg,zero,total])

    final = pd.DataFrame.from_dict(checks).T
    final.columns = ['nans','negatives','zeroes','totals']
    final.to_csv(i / 'checks.csv')

def check_growth_factors():
    def dvec_segment_summary(dvec: nd.DVector):
        for name in dvec.segmentation.segment_names:
            arr = dvec._data[name]
            yield (
                *name.split("_"),
                np.sum(np.isnan(arr)),
                np.sum(arr == 0),
                np.sum(arr < 0),
                np.sum(arr > 0),
                arr.size,
            )
    folder = pathlib.Path(r"T:\MidMITs Demand\Forecasting\miham\iter9.7-COVID\TEMPro Growth Factors")
    output_folder = pathlib.Path("DO_NOT_COMMIT/Forecasting checks")
    output_folder.mkdir(exist_ok=True)
    print(f"Created: {output_folder}")
    for file in tqdm(list(folder.iterdir()), dynamic_ncols=True):
        data = nd.DVector.load(file)
        with open(output_folder / (file.stem + ".csv"), "wt", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                data.segmentation.naming_order + ["# NaNs", "# 0", "# < 0", "# > 0", "Size"]
            )
            writer.writerows(dvec_segment_summary(data))