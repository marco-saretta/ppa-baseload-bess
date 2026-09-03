"""Project filesystem paths.

All paths are absolute so they resolve correctly regardless of the current
working directory (Hydra may change it for a run).
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
