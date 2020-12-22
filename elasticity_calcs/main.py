# -*- coding: utf-8 -*-
"""
    Module for running the elasticity model from command line.
"""

##### IMPORTS #####
# Standard imports
import sys
from pathlib import Path

# Third party imports

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from elasticity_calcs.elasticity_model import ElasticityModel


##### FUNCTIONS #####
def get_inputs():
    # TODO Change hardcoded paths to config file
    input_folders = {
        "elasticity": Path("C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/elasticities"),
        "translation": Path("C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/translation"),
        "rail_demand": Path(
            "C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/rail demand"
        ),
        "car_demand": Path(
            "C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/car demand"
        ),
        "rail_costs": Path(
            "C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/rail costs"
        ),
        "car_costs": Path("C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/car costs"),
    }
    output_folder = Path("C:/WSP_Projects/TfN EFS/02 Delivery/03 - D3/Test/outputs")
    years = ["2018"]
    return input_folders, output_folder, years


def main():
    input_folders, output_folder, years = get_inputs()

    elast_model = ElasticityModel(input_folders, output_folder, years)
    elast_model.apply_all()


##### MAIN #####
if __name__ == "__main__":
    working_dir = Path.cwd().parent
    print(working_dir)
    sys.path.append(working_dir)
    main()
