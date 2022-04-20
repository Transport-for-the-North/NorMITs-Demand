
import os
from normits_demand.utils import cjtw_processing

if __name__ == '__main__':
    
    model_name = 'Nelum'
    model_folder = 'I:/NorMITs Synthesiser/Nelum/Model Zone Lookups'
    out_drive = 'I:\\'
    out_path = os.path.join(out_drive,
                            r'NorMITs Synthesiser\Nelum\iter5\cjtw working')

    co = cjtw_processing.CjtwTranslator(model_name=model_name,
                                        model_folder=model_folder)

    base_cjtw = co.build_base_cjtw()
    future_cjtw = co.cjtw_to_future_year(base_cjtw,
                                         target_year=2018)
    cjtw_to_model_zone = co.cjtw_to_model_zone(future_cjtw)

    # Write out
    msoa_path = os.path.join(out_path, '2018 msoa')
    co.export_by_segment(future_cjtw,
                         model_zoning='msoa',
                         year=2018,
                         out_path=msoa_path)
    nelum_path = os.path.join(out_path, '2018 nelum prior method')
    co.export_by_segment(cjtw_to_model_zone,
                         model_zoning='nelum',
                         year=2018,
                         out_path=nelum_path)