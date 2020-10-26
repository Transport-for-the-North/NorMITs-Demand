# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:27:28 2020

@author: Sneezy
"""

from functools import reduce
from typing import List

import numpy as np
import pandas as pd

class SectorReporter:
    """
    Sector reporting class for use in NorMITs Demand framework.
    """
    # defaults
    default_zone_system_name = str
    default_zone_system = pd.DataFrame
    default_sector_grouping = pd.DataFrame
    
    def __init__(self,
                 default_zone_system: str = "MSOA",
                 default_zone_file: str = "Y:/NorMITs Synthesiser/Repo/Normits-Utils/zone_groupings/msoa_zones.csv",
                 default_sector_grouping_file: str = "Y:/NorMITs Synthesiser/Repo/Normits-Utils/zone_groupings/lad_msoa_grouping.csv"
                 ):
        """
        Initialises the sector reporting class.
        
        Parameters
        ----------
        default_zone_system:
            The default zoning system title. Unused but necessary for potential
            future improvements or changes.
            Default input is: "MSOA".
            Possible input is any string.
            
        default_zone_file:
            The default zone file which is used to read in the full list of
            model zones. Used for zone checking and calculation of any
            missing sectors.
            Possible input is any file location.
            This file is required to have the column: model_zone_id
            
        default_sector_grouping_file:
            The default zone file which is used to read in the full list of
            sector groups and model zones. Used to determine zone groupings
            for sector total calculation.
            Possible input is any file location.
            This file is required to have the columns: grouping_id, model_zone_id
        """
        self.default_zone_system_name = default_zone_system
        self.default_zone_system = pd.read_csv(default_zone_file)[[
                "model_zone_id"
                ]]
        self.default_sector_grouping = pd.read_csv(default_sector_grouping_file)[[
                "grouping_id",
                "model_zone_id"
                ]]
        
    def calculate_sector_totals(self,
                                calculating_dataframe: pd.DataFrame,
                                grouping_metric_columns: List[str] = None,
                                zone_system_name: str = None,
                                zone_system_file: str = None,
                                sector_grouping_file: str = None
                                ) -> pd.DataFrame:
        """
        Calculates the sector totals off the given parameters.
        
        Parameters
        ----------
        calculating_dataframe:
            This is the dataframe from which the sector totals will be
            calculated. This dataframe needs at least a "model_zone_id"
            column but any additional non-categoric columns can be included.
            No default input.
            Possible input is any Pandas Dataframe with a "model_zone_id"
            column.
            
        grouping_metric_columns:
            The columns to be grouped into sector totals.
            Default input is: None. If this default input is used then
            all columns (except "model_zone_id") are selected.
            Possible input is any list of strings.
            THESE ARE THE COLUMNS WE KEEP

        zone_system_name:
            The name of the zone system for this data set.
            Default input is: None. If this default input is used then
            the zone system name is set to the default for this object.
            Possible input is any file location.
            
        zone_system_file:
            The  zone file which is used to read in the full list of
            model zones. Used for zone checking and calculation of any
            missing sectors.
            Default input is: None. If this default input is used then the
            zone system is set to the default for this boject.
            Possible input is any file location.
            This file is required to have the columns: model_zone_id
            
        sector_grouping_file:
            The default zone file which is used to read in the full list of
            sector groups and model zones. Used to determine zone groupings
            for sector total calculation.
            Default input is: None. If this default input is used then
            the sector grouping is set to the default for this object.
            Possible input is any file location.
            This file is required to have the columns: grouping_id, model_zone_id
            
        Return
        ----------
        sector_totals:
            The returned sector totals in Pandas Dataframe form. This will
            include a summed total of each column in either grouping_metric_columns
            or all the non-model zone ID columns depending on what is input in
            grouping_metric_columns.
            
        Future Improvements
        ----------
        Pass different total values. Preserve, as an example, purposes or modes.
        Include zone checks to ensure everything matches up.
        Include zone conversion from default zone to this zone system.
        Different aggregation / total methods. Average or median are examples.
        """
        calculating_dataframe = calculating_dataframe.copy()
        
        if (zone_system_name != None):
            # do nothing
            print("Changing zone system name...")
        else:
            print("Not changing zone system name...")
            zone_system_name = self.default_zone_system_name
        
        if (zone_system_file != None):
            # read in file
            print("Changing zone system...")
            zone_system = pd.read_csv(zone_system_file)
        else:
            print("Not changing zone system...")
            zone_system = self.default_zone_system.copy()
            
        if (sector_grouping_file != None):
            # read in file
            print("Changing sector grouping...")
            sector_grouping = pd.read_csv(sector_grouping_file)
        else:
            print("Not changing sector grouping...")
            sector_grouping = self.default_sector_grouping.copy()
            
        sector_grouping_zones = sector_grouping["model_zone_id"].unique()
            
        if (grouping_metric_columns == None):
            grouping_metric_columns = calculating_dataframe.columns[1:]
            
        groupings = sector_grouping["grouping_id"].unique()
        grouping_dataframe_columns = ["grouping_id"]
        grouping_dataframe_columns.extend(grouping_metric_columns)
        sector_totals = pd.DataFrame(
                columns = grouping_dataframe_columns
                )
        
        for group in groupings:
            zones = sector_grouping[
                sector_grouping["grouping_id"] == group
            ]["model_zone_id"].values
            calculating_dataframe_mask = calculating_dataframe["model_zone_id"].isin(zones)
            new_grouping_dataframe = pd.DataFrame({"grouping_id": [group]})
    
            for metric in grouping_metric_columns:
                new_grouping_dataframe[metric] = calculating_dataframe[
                    calculating_dataframe_mask
                ].sum()[metric]
                
            print(new_grouping_dataframe)
            sector_totals = sector_totals.append(new_grouping_dataframe)
            
        return sector_totals
