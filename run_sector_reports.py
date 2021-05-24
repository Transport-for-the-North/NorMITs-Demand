"""
Sector reports script
just to show implementation of method
Will need to be integrated into demand reporting framework
"""

import os

# local imports
from normits_demand.reports import sector_report

if __name__ == '__main__':

    model_name = 'noham'
    run_folder = 'I:/NorMITs Demand/%s/EFS' % model_name
    scenarios = ['SC01_JAM', 'SC02_PP', 'SC03_DD', 'SC04_UZC']
    out_folder = 'C:/Users/%s/Documents' % os.getlogin()

    folder_list = list()
    for sc in scenarios:
        folder_list.append(
            os.path.join(
                run_folder, 'iter3f/%s/Matrices/Compiled OD Matrices' % sc))

    # run reporter
    for folder in folder_list:
        print('Running sector reports for %s' % folder)
        sr = sector_report.SectorReporter(target_folder=folder,
                                          model_name=model_name,
                                          output_folder=out_folder)
        reports = sr.sector_report()










