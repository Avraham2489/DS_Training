"""Shared runtime configuration for the regression exercise."""

import logging
from pathlib import Path
import sys
import warnings


warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

LOGGER = logging.getLogger(name="regression_exercise")
RANDOM_SEED = 202_607_27
EXERCISE_DIRECTORY = Path(__file__).resolve().parent
RESOURCE_DIRECTORY = EXERCISE_DIRECTORY / "resources"
