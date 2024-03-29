# -*- coding: utf-8 -*-
"""
    Module for testing functions in demand_utilities.utils module.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path

# Third party imports
import pandas as pd

# Local imports
from normits_demand.utils.general import get_zone_translation


##### FUNCTIONS #####
def test_get_zone_translation(tmp_path: Path):
    """Test the `get_zone_translation` function returns the expected DataFrame.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    from_zone, to_zone = "noham", "norms"
    name = "{}_zone_id"
    translation = pd.DataFrame(
        {name.format(from_zone): [1, 2, 3], name.format(to_zone): [1, 1, 2]}
    )
    translation.to_csv(tmp_path / f"{from_zone}_to_{to_zone}.csv", index=False)

    pd.testing.assert_frame_equal(
        translation,
        get_zone_translation(
            tmp_path, from_zone, to_zone, return_dataframe=True
        ),
        check_dtype=False,
    )
