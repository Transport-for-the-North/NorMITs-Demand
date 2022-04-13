"""
Sector reports script
just to show implementation of method
Will need to be integrated into demand reporting framework
"""

import os

# local imports
from normits_demand.reports import old_sector_report

import pathlib

if __name__ == '__main__':

    model_name = 'noham'
    run_folder = 'I:/NorMITs Demand/%s/EFS' % model_name
    # scenarios = ['SC01_JAM', 'SC02_PP', 'SC03_DD', 'SC04_UZC']
    scenarios = ['SC02_PP']
    out_folder = 'C:/Users/genie/Documents/Sector Reports'

    folder_list = list()
    for sc in scenarios:
        folder_list.append(
            os.path.join(
                run_folder, 'iter3i/%s/Matrices/Compiled OD Matrices' % sc))

    # run reporter
    for folder in folder_list:
        print('Running sector reports for %s' % folder)
        sr = old_sector_report.SectorReporter(target_folder=folder,
                                          model_name=model_name,
                                          output_folder=out_folder)
        reports = sr.sector_report()

        # Unpack and export sector reports
        out_reps = dict()
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
                    write_folder = os.path.join(out_folder,
                                                fname)
                    print('exporting %s to %s' % (fname, write_folder))

                    # Build out_reps for compilation
                    out_reps.update({str(fname): dat})

                    dat.to_csv(write_folder)

        # Compile 24hr Norms internal PA
        commute = dict()
        business = dict()
        other = dict()
        for csv, dat in out_reps.items():
            if 'W' in csv and 'Int' in csv and '2050' in csv:
                commute.update({csv: dat})
            elif 'EB' in csv and 'Int' in csv and '2050' in csv:
                business.update({csv: dat})
            elif 'O' in csv and 'Int' in csv and '2050' in csv:
                other.update({csv: dat})

        commute_out = list()
        for csv, dat in commute.items():
            commute_out.append(dat)
        commute_out = sum(commute_out)

        business_out = list()
        for csv, dat in business.items():
            business_out.append(dat)
        business_out = sum(business_out)

        other_out = list()
        for csv, dat in other.items():
            other_out.append(dat)
        other_out = sum(other_out)

        commute_out.to_csv(os.path.join(out_folder, 'commute_int_24hr.csv'))
        business_out.to_csv(os.path.join(out_folder, 'business_int_24hr.csv'))
        other_out.to_csv(os.path.join(out_folder, 'other_int_24hr.csv'))

        # Compile 12hr Noham internal
        commute = dict()
        business = dict()
        other = dict()
        for csv, dat in out_reps.items():
            if 'commute' in csv and '2018' in csv:
                if 'tp4' not in csv:
                    commute.update({csv: dat})
            elif 'business' in csv and '2018' in csv:
                if 'tp4' not in csv:
                    business.update({csv: dat})
            elif 'other' in csv and '2018' in csv:
                if 'tp4' not in csv:
                    other.update({csv: dat})

        commute_out = list()
        for csv, dat in commute.items():
            commute_out.append(dat)
        commute_out = sum(commute_out)

        business_out = list()
        for csv, dat in business.items():
            business_out.append(dat)
        business_out = sum(business_out)

        other_out = list()
        for csv, dat in other.items():
            other_out.append(dat)
        other_out = sum(other_out)

        commute_out.to_csv(os.path.join(out_folder, 'commute_2018_12hr.csv'))
        business_out.to_csv(os.path.join(out_folder, 'business_2018_12hr.csv'))
        other_out.to_csv(os.path.join(out_folder, 'other_2018_12hr.csv'))















