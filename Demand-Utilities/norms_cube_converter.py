# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:31:46 2020

@author: Sneezy
"""
from typing import List

import pandas as pd
import numpy as np

### NoRMS Cube Input Converter
class NoRMS_Cube_Converter:
    """
    #TODO
    """
    purpose_change_dictionary_from = str
    purpose_change_dictionary_to = str
    purpose_change_dictionary = dict()
    
    def __init__(self,
                 purpose_change_dictionary_from: str = "Synthesiser",
                 purpose_change_dictionary_to: str = "NoRMS",
                 purpose_change_dictionary: dict = {
                         1: 1,
                         2: 2,
                         3: 3,
                         4: 3,
                         5: 3,
                         6: 3,
                         7: 3,
                         8: 3
                         } # Synthesiser Purposes by Default
                 ):
        """
        #TODO
        """
        self.purpose_change_dictionary_from = purpose_change_dictionary_from
        self.purpose_change_dictionary_to = purpose_change_dictionary_to
        self.purpose_change_dictionary = purpose_change_dictionary
    
    def retrieve_internal_distribution(self,
                                       distribution: pd.DataFrame,
                                       internal_zones: List[int],
                                       production_zone_column: str = "p_zone",
                                       attraction_zone_column: str = "a_zone",
                                       distribution_column: str = "dt"
                                       ) -> pd.DataFrame:
        """
        #TODO
        """
        internal_distribution = distribution.copy()
        
        internal_mask = (
                (
                        internal_distribution[
                                production_zone_column
                                ].isin(
                                internal_zones
                                )
                )
                &
                (
                        internal_distribution[
                                attraction_zone_column
                                ].isin(
                                internal_zones
                                )
                )
                )
                
        internal_distribution.loc[
                ~internal_mask,
                distribution_column
                ] = 0      
        
        return internal_distribution
    
    def retrieve_external_distribution(self,
                                       distribution: pd.DataFrame,
                                       internal_zones: List[int],
                                       production_zone_column: str = "p_zone",
                                       attraction_zone_column: str = "a_zone",
                                       distribution_column: str = "dt"
                                       ) -> pd.DataFrame:
        """
        #TODO
        """
        external_distribution = distribution.copy()
        
        internal_mask = (
                (
                        external_distribution[
                                production_zone_column
                                ].isin(
                                internal_zones
                                )
                )
                &
                (
                        external_distribution[
                                attraction_zone_column
                                ].isin(
                                internal_zones
                                )
                )
                )
                
        external_distribution.loc[
                internal_mask,
                distribution_column
                ] = 0      
        
        return external_distribution
    
    def purpose_converter(self,
                        dataframe: pd.DataFrame,
                        grouping_list: list = None
                        ) -> pd.DataFrame:
        """
        Converts purposes from one dataframe to another.
        Functionally just renames the purposes to the new purpose types.
        Aggregation performed by aggregate_dataframe later on in run
        sequence.

        Parameters
        ----------
        dataframe:
            A pandas dataframe with a "purpose_id" column.

        Return
        ----------
        converted_frame:
            A pandas dataframe with the updated purposes.
        """
        change_dictionary = self.purpose_change_dictionary
        dataframe_purpose_ids = dataframe.purpose_id.unique()

        converted_frame = pd.DataFrame
        first_iteration = True

        for new_purpose_id in change_dictionary.keys():
            for old_purpose_id in change_dictionary[new_purpose_id]:
                if (old_purpose_id in dataframe_purpose_ids):
                    # for bug testing
                    print("Changing " + str(old_purpose_id) + " to "
                          + str(new_purpose_id) + ".")

                    chopped_frame = dataframe[
                            dataframe["purpose_id"] == old_purpose_id
                            ]
                    
                    chopped_frame.loc[
                            chopped_frame["purpose_id"] == old_purpose_id,
                            ["purpose_id"]
                            ] = new_purpose_id

                    # if this is our first iteration
                    if (first_iteration):
                        # then the converted frame does not need joining
                        converted_frame = chopped_frame
                        first_iteration = False

                    else:
                        # else we need to merge the two frames
                        converted_frame = pd.concat(

                                [
                                        converted_frame,
                                        chopped_frame
                                        ]
                                )

        if (grouping_list != None):
            converted_frame = converted_frame.groupby(
                    grouping_list,
                    as_index = False
                    ).sum()

        return converted_frame