import pandas as pd
import pathlib
import os

def compare_folders(pre_folder: pathlib.Path, post_folder: pathlib.Path):
    checks = {}
    for file in os.listdir(pre_folder):
        inner={}
        pre = pd.read_csv(pre_folder / file)
        try:
            post = pd.read_csv(post_folder / file)
        except FileNotFoundError:
            post = pd.read_csv(post_folder / (file + '.bz2'))
        diff = post - pre
        inner['max'] = diff.max().max()
        inner['min'] = diff.min().min()
        inner['total'] = diff.sum().sum()
        checks[file] = inner
    return pd.DataFrame.from_dict(checks, orient='index')

if __name__ == '__main__':
    compiled_od = compare_folders(pathlib.Path(r"E:\noham-to-nocarb\comparison\compiled_od\pre"),
                    pathlib.Path(r"E:\noham-to-nocarb\comparison\compiled_od\post"))

    compiled_od.to_csv(r"E:\noham-to-nocarb\comparison\compiled_od\comp.csv")

    # od = compare_folders(pathlib.Path(r"E:\noham-to-nocarb\comparison\od\pre"),
    #                 pathlib.Path(r"E:\noham-to-nocarb\comparison\od\post"))

    # pa = compare_folders(pathlib.Path(r"E:\noham-to-nocarb\comparison\pa\pre"),
    #                      pathlib.Path(r"E:\noham-to-nocarb\comparison\pa\post"))

    print('done')