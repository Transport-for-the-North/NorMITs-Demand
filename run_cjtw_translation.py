
import os
import itertools
from normits_demand.utils import cjtw_processing

if __name__ == '__main__':
    
    model_name = 'Nelum'
    model_folder = 'I:/NorMITs Synthesiser/Nelum/Model Zone Lookups'
    out_drive = 'C:\\'
    out_path = os.path.join(out_drive,
                            'Users',
                            os.getlogin(),
                            'Documents')

    co = cjtw_processing.CjtwTranslator(model_name=model_name,
                                        model_folder=model_folder)

    nelum_cjtw = co.cjtw_to_model_zone(target_year=2018)

    # Build export iterator
    # TODO: Make customisable segments
    unq_tp = nelum_cjtw['TimePeriod'].unique()
    unq_m = nelum_cjtw['mode'].unique()

    for tp, m in itertools.product(unq_tp, unq_m):

        out_sub = nelum_cjtw.copy()
        # Filter
        out_sub = out_sub[out_sub['TimePeriod'] == tp]
        out_sub = out_sub[out_sub['mode'] == tp]

        # Path
        path = 'nelum_cjtw_yr2018_p1_m%d_tp%d' % (m, tp)

        # Out
        out_sub.to_csv(os.path.join(
            out_path, path + '.csv'), index=False)
