"""Shared runtime configuration for the clustering and PCA recitation."""

import logging
import sys
import warnings


warnings.filterwarnings(action="ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

LOGGER = logging.getLogger(name="clustering_pca_recitation")
RANDOM_SEED = 42
