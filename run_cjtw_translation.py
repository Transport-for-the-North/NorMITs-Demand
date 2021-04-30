
import os
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
    for path, dat in nelum_cjtw.items():
        dat.to_csv(os.path.join(
            out_path, path + '.csv'), index=False)
