"""Shared deterministic runtime configuration for the recitation."""

import logging
import sys
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

LOGGER = logging.getLogger(name="gradient_descent_recitation")
RANDOM_SEED = 202_607_27
