"""Utilities for the supervised-learning exercise notebook."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def make_local_weight_function(
    *, reference_sample: NDArray[np.floating],
) -> Callable[..., NDArray[np.floating]]:
    """Create an inverse-Euclidean-distance locality weighting function.

    Args:
        reference_sample: One feature vector that defines the local neighborhood.

    Returns:
        A keyword-only function that assigns ``1 / (distance + 1)`` to each row.
    """
    reference_array = np.asarray(a=reference_sample, dtype=np.float64).reshape(1, -1)

    def calculate_local_weights(
        *, samples: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Calculate locality weights for one sample or a matrix of samples.

        Args:
            samples: One feature vector or a two-dimensional sample matrix.

        Returns:
            One locality weight per supplied sample.
        """
        sample_array = np.asarray(a=samples, dtype=np.float64)
        sample_matrix = sample_array.reshape(1, -1) if sample_array.ndim == 1 else sample_array
        if sample_matrix.shape[1] != reference_array.shape[1]:
            raise ValueError(
                "Samples and reference_sample must contain the same number of features."
            )
        distances = np.linalg.norm(x=sample_matrix - reference_array, axis=1)
        return 1.0 / (distances + 1.0)

    return calculate_local_weights
