
import os

from normits_demand.utils import nts_processing as nts

if __name__ == '__main__':
    """
    Run trip length extraction from NTS
    """

    # TODO: path and smart search should be in constants
    _TLB_FOLDER = 'I:/NorMITs Demand/import/trip_length_distributions'
    _NTS_IMPORT = 'I:/NTS/classified builds/cb_tfn.csv'

    run_another = True
    while run_another:
        extract = nts.NTSTripLengthBuilder(tlb_folder=_TLB_FOLDER,
                                           nts_import=_NTS_IMPORT)

        dat = extract.run_tlb_lookups()

        if input('Run another y/n').lower() == 'n':
            run_another = False
