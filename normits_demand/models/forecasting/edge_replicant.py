# -*- coding: utf-8 -*-
"""EDGE forecasting model."""
# ## IMPORTS ## #
# Standard imports

# Third party imports

# Local imports
# pylint: disable=import-error,wrong-import-position
from normits_demand import logging as nd_log
from normits_demand.models.forecasting import forecast_cnfg
from normits_demand.models.forecasting import edge_forecast
# pylint: enable=import-error,wrong-import-position

# ## CONSTANTS ## #
LOG = nd_log.get_logger(__name__)

# ## CLASSES ## #

# ## FUNCTIONS ## #
def main(params: forecast_cnfg.EDGEParameters) -> None:
    edge_forecast.RunEDGEGrowth(params)
    raise NotImplementedError("Not yet implemented main function for EDGE model")
    
