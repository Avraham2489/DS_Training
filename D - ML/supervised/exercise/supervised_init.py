"""Shared initialization for the supervised-learning exercise."""

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("exercise_4_supervised_learning")

RANDOM_SEED = 42

