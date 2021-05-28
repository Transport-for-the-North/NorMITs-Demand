"""
Sector reports script
just to show implementation of method
Will need to be integrated into demand reporting framework
"""

import os

# local imports
from normits_demand.reports import sector_report

import pathlib

if __name__ == '__main__':

    model_name = 'noham'
    run_folder = 'I:/NorMITs Demand/%s/EFS' % model_name
    # scenarios = ['SC01_JAM', 'SC02_PP', 'SC03_DD', 'SC04_UZC']
    scenarios = ['SC04_UZC']
    out_folder = 'E:/'

    folder_list = list()
    for sc in scenarios:
        folder_list.append(
            os.path.join(
                run_folder, 'iter3g/%s/Matrices/Aggregated PA Matrices' % sc))

    # run reporter
    for folder in folder_list:
        print('Running sector reports for %s' % folder)
        sr = sector_report.SectorReporter(target_folder=folder,
                                          model_name=model_name,
                                          output_folder=out_folder)
        reports = sr.sector_report()

        # Unpack and export sector reports
        # TODO: This dictionary unpacker is not very clean
        for mat, report_dict in reports.items():
            print('unpacking reports for %s' % mat)
            for r_name, dat in report_dict.items():
                if r_name == 'ca_sectors':
                    # Hard path to output folder
                    # out_folder = folder.replace(
                    #     'Matrices/Compiled OD Matrices',
                    #     'Reports/Sector Reports')
                    fname = pathlib.Path(mat)
                    fname = fname.parent / (fname.stem + '_ca_sectors' + fname.suffix)
                    out_folder = os.path.join(out_folder,
                                              fname)
                    print('exporting %s to %s' % (fname, out_folder))

                    dat.to_csv(out_folder)











