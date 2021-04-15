
import os
from normits_demand.utils import cjtw_processing as cjtw

if __name__ == '__main__':
    
    model_name = 'Nelum'
    model_folder = 'I:/NorMITs Synthesiser/Nelum/Model Zone Lookups'

    nel_to_cjtw = cjtw.CjtwTranslator(model_name=model_name,
                                      model_folder=model_folder)

    nelum_cjtw, audits = nel_to_cjtw.cjtw_to_zone_translation(write=False)