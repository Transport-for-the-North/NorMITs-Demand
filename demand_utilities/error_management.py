# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:43:23 2019

@author: RobertFarrar
"""

from typing import List, Any
    
def check_values(check_list: List[Any],
                 input_value: Any,
                 custom_string: str = None
                 ) -> bool:
    """
    #TODO
    """
    if (input_value in check_list):
        return True
    else:
        if (custom_string == None):
            raise ValueError("ValueError du to input value " + input_value
                         + " not being in acceptable range of "
                         + str(check_list) + ".")
        else:
            raise ValueError(custom_string)
    
def check_type(expected_type: type,
               input_value: Any,
               custom_string: str = None
               ) -> bool:
    """
    #TODO
    """
    if (expected_type == type(input_value)):
        return True
    else:
        if (custom_string == None):
            raise TypeError("TypeError du to incorrect input value. "
                            + "Type of input is: " + str(type(input_value)) + ". "
                            + "Expected type: " + str(expected_type) + ".")
        else:
            raise TypeError(custom_string)