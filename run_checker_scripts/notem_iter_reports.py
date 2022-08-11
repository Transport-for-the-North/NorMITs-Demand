# -*- coding: utf-8 -*-
"""
Created on: 29/03/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import sys

# Third Party
import pandas as pd
import tqdm

# Local Imports
sys.path.append('..')
import normits_demand as nd

from normits_demand.pathing import NoTEMExportPaths


YEAR = 2018
SCENARIO = nd.Scenario.SC01_JAM
NOTEM_PATH = r"I:\NorMITs Demand\NoTEM"
HB_COMPARE_SEG = nd.get_segmentation_level('hb_p_m_tp_week')
NHB_COMPARE_SEG = nd.get_segmentation_level('nhb_p_m_tp_week')

COMPARE_ITERS = [
    '9.3',
    '9.4',
    '9.5',
    '9.6',
    '9.7',
]

OUTPUT_PATH = 'E:/temp/NoTEM compare report.csv'


def main():

    pbar = tqdm.tqdm(
        desc='Generating reports',
        total=len(COMPARE_ITERS) * 4,
    )

    # Get the data for each iteration
    report_list = list()
    for iter_name in COMPARE_ITERS:

        # Generate the export paths
        export_paths = NoTEMExportPaths(
            path_years=[YEAR],
            scenario=SCENARIO,
            iteration_name=iter_name,
            export_home=NOTEM_PATH,
        )

        # Get the data for each type of trip end
        dvec_paths = {
            'hb_production': export_paths.hb_production.export_paths.notem_segmented[YEAR],
            'hb_attraction': export_paths.hb_attraction.export_paths.notem_segmented[YEAR],
            'nhb_production': export_paths.nhb_production.export_paths.notem_segmented[YEAR],
            'nhb_attraction': export_paths.nhb_attraction.export_paths.notem_segmented[YEAR],
        }
        for name, trip_end_path in dvec_paths.items():
            if name[:2] == 'hb':
                compare_seg = HB_COMPARE_SEG
            else:
                compare_seg = NHB_COMPARE_SEG

            # Aggregate and convert
            dvec = nd.DVector.load(trip_end_path)
            dvec = dvec.aggregate(compare_seg)
            dvec = dvec.convert_time_format('avg_day')
            dvec = dvec.translate_zoning(nd.get_zoning_system('ca_sector_2020'))

            # Convert to df and add to report list
            df = dvec.to_df()
            df['iter'] = iter_name
            df['trip_end'] = name

            report_list.append(df)

            pbar.update(1)

    pbar.close()

    # Concatenate all data and output
    full_report = pd.concat(report_list, ignore_index=True)
    full_report.to_csv(OUTPUT_PATH, index=False)


if __name__ == '__main__':
    main()
