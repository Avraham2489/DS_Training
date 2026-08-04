"""Shared initialization for the SVM recitation notebook."""

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("svm_recitation")

RANDOM_SEED = 42

