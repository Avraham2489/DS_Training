"""Tests for supervised-learning notebook utilities."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

UTILITY_PATH = (
    Path(__file__).parents[1]
    / "D - ML"
    / "supervised"
    / "exercise"
    / "supervised_utils.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    name="supervised_utils", location=UTILITY_PATH
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Unable to load supervised utilities from {UTILITY_PATH}")
SUPERVISED_UTILS = importlib.util.module_from_spec(spec=MODULE_SPEC)
MODULE_SPEC.loader.exec_module(module=SUPERVISED_UTILS)


def test_make_local_weight_function_returns_expected_weights() -> None:
    """The reference has unit weight and farther rows receive lower weights."""
    calculate_weights = SUPERVISED_UTILS.make_local_weight_function(
        reference_sample=np.array(object=[0.0, 0.0])
    )

    weights = calculate_weights(
        samples=np.array(object=[[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    )

    np.testing.assert_allclose(actual=weights, desired=np.array([1.0, 1.0 / 6.0, 1.0 / 11.0]))


def test_make_local_weight_function_rejects_incompatible_shapes() -> None:
    """A mismatched feature count raises an explicit error."""
    calculate_weights = SUPERVISED_UTILS.make_local_weight_function(
        reference_sample=np.array(object=[0.0, 0.0])
    )

    with pytest.raises(
        expected_exception=ValueError,
        match="same number of features",
    ):
        calculate_weights(samples=np.array(object=[[0.0, 0.0, 0.0]]))
