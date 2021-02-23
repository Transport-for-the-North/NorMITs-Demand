"""
    Configuration file ran by pytest.
"""

import sys
from pathlib import Path

# Add root folder to path so tests can import normits_demand
sys.path.append(str(Path.cwd()))
