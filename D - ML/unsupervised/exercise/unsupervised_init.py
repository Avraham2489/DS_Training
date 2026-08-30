"""Shared runtime configuration for the unsupervised-learning exercise."""

from datetime import datetime
import logging
from pathlib import Path
import sys
import warnings


warnings.filterwarnings(action="ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

LOGGER = logging.getLogger(name="unsupervised_exercise")
RANDOM_SEED = 42
EXERCISE_DIRECTORY = Path(__file__).resolve().parent
IMAGE_DIRECTORY = EXERCISE_DIRECTORY.parent / "images"

current_time = datetime.now()
OUTPUT_DIRECTORY = (
    EXERCISE_DIRECTORY
    / "outputs"
    / f"clustering_exercise_{current_time:%Y%m%d_%H%M%S}"
)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
