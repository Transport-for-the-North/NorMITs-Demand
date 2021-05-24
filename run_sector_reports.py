"""
Sector reports script
just to show implementation of method
Will need to be integrated into demand reporting framework
"""

import os

# local imports
from normits_demand.reports import sector_report

if __name__ == '__main__':

    tf = r'I:\NorMITs Demand\norms\EFS\iter3g\SC04_UZC\Matrices\Aggregated PA Matrices'
    mn = 'norms'
    out_folder = 'C:/Users/%s/Documents' % os.getlogin()

    sr = sector_report.SectorReporter(target_folder=tf,
                                      model_name=mn,
                                      output_folder=out_folder)







