
import os
import pandas as pd

from normits_demand.utils import utils as nup
from normits_demand.matrices import translate_matrices as tm

if __name__ == '__main__':
    import_folder = (r'I:\NorMITs Demand\noham\v0.3-EFS_Output\NTEM\iter3f\Matrices\OD Matrices')
    export_folder = (r'I:\NorMITs Synthesiser\Nelum\iter2\Outputs\From Home Matrices')
    nel_export_folder = (r'I:\NorMITs Synthesiser\Nelum\iter2\Outputs\From Home Matrices NELUM')
    compile_params = 'fhma.csv'

    translation_lookup_folder = ('I:/NorMITs Synthesiser/Zone Translation/Export')

    start_zone_model_folder = (r'I:\NorMITs Synthesiser\Noham\Model Zone Lookups')
    end_zone_model_folder = (r'I:\NorMITs Synthesiser\Nelum\Model Zone Lookups')

    start_zoning_system = 'Noham'
    end_zoning_system = 'Nelum'

    od_in = nup.compile_od(
        od_folder=import_folder,
        write_folder=export_folder,
        compile_param_path=os.path.join(
            export_folder, compile_params),
        build_factor_pickle=False)

    nelum_out = tm.translate_matrices(
        start_zoning_system,
        end_zoning_system,
        translation_lookup_folder,
        import_folder=export_folder,
        export_folder=nel_export_folder,
        start_zone_model_folder=start_zone_model_folder,
        translation_type='pop',  # default to pop
        mat_format='od',
        before_tld=True,
        after_tld=False,
        export=True
    )


