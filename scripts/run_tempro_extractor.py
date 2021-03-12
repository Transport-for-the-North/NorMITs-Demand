
import pandas as pd

import normits_demand.utils.tempro_extractor as te

if __name__ == '__main__':
    co = te.TemproParser()
    co_out = co.get_co_future_share()
    co_out[0].to_csv('C:/Users/Genie/Documents/ca_factors.csv', index=False)