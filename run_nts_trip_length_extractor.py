
import os

from normits_demand.utils import nts_processing as nts

if __name__ == '__main__':
    """
    Run trip length extraction from NTS
    """

    # TODO: path and smart search should be in constants
    _TLB_FOLDER = 'I:/NorMITs Synthesiser/import/trip_length_bands'
    _NTS_IMPORT = 'Y:/NTS/import/classified builds/cb_tfn.csv'

    xtract = nts.NTSTripLengthBuilder(tlb_folder=_TLB_FOLDER,
                                      nts_import=_NTS_IMPORT)

    dat = xtract.run_tlb_lookups()