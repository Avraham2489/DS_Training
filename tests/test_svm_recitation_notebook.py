"""Tests for the SVM implementations embedded in the recitation notebook."""

from pathlib import Path
from typing import Any

import matplotlib
import nbformat
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "D - ML"
    / "supervised"
    / "recitation"
    / "svm - recitation.ipynb"
)
DATA_DIRECTORY = NOTEBOOK_PATH.parent.parent


def load_function_namespace() -> dict[str, Any]:
    """Execute only the notebook cells that define tested functions.

    Returns:
        Namespace containing the notebook's SVM and plotting functions.
    """
    notebook = nbformat.read(fp=NOTEBOOK_PATH, as_version=4)
    definition_prefixes = (
        "def predict(",
        "def plot_data(",
        "def gaussian_kernel_function(",
        "def regularized_linear_svm(",
    )
    namespace: dict[str, Any] = {
        "np": np,
        "plt": plt,
        "Axes": Axes,
        "NDArray": NDArray,
    }
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.startswith(definition_prefixes):
            exec(compile(cell.source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_predict_and_linear_svm_fit_real_linear_dataset() -> None:
    """The implemented gradient method separates the provided linear data."""
    namespace = load_function_namespace()
    linear_data = np.loadtxt(
        fname=DATA_DIRECTORY / "svm_dataset1.csv",
        delimiter=",",
        skiprows=1,
    )
    features = linear_data[:, :2]
    labels = linear_data[:, 2]

    weights, bias, loss_history = namespace["linear_svm"](
        features=features,
        labels=labels,
        epochs=2_000,
        learning_rate=0.05,
    )
    predictions = (
        namespace["predict"](
            features=features,
            weights=weights,
            bias=bias,
        )
        >= 0.0
    ).astype(np.float64)

    assert float(np.mean(a=predictions == labels)) >= 0.98
    assert loss_history[-1] < loss_history[0]


def test_gaussian_kernel_is_symmetric_with_unit_diagonal() -> None:
    """The RBF transform has its defining properties on real samples."""
    namespace = load_function_namespace()
    nonlinear_data = np.loadtxt(
        fname=DATA_DIRECTORY / "svm_dataset2.csv",
        delimiter=",",
        skiprows=1,
    )
    features = nonlinear_data[:25, :2]

    kernel_matrix = namespace["gaussian_kernel_function"](
        features=features,
        landmark_features=features,
        sigma=0.1,
    )

    np.testing.assert_allclose(actual=kernel_matrix, desired=kernel_matrix.T)
    np.testing.assert_allclose(
        actual=np.diag(v=kernel_matrix),
        desired=np.ones(shape=len(features)),
    )
    assert np.all(a=(kernel_matrix >= 0.0) & (kernel_matrix <= 1.0))


def test_regularization_reduces_weight_norm_on_real_data() -> None:
    """The bonus implementation applies measurable L2 shrinkage."""
    namespace = load_function_namespace()
    linear_data = np.loadtxt(
        fname=DATA_DIRECTORY / "svm_dataset1.csv",
        delimiter=",",
        skiprows=1,
    )
    features = linear_data[:, :2]
    labels = linear_data[:, 2]

    unregularized_weights, _, _ = namespace["linear_svm"](
        features=features,
        labels=labels,
        epochs=2_000,
        learning_rate=0.05,
    )
    regularized_weights, _, _ = namespace["regularized_linear_svm"](
        features=features,
        labels=labels,
        regularization_strength=0.3,
        epochs=2_000,
        learning_rate=0.05,
    )

    assert np.linalg.norm(x=regularized_weights) < np.linalg.norm(
        x=unregularized_weights
    )


def test_plot_data_draws_both_real_classes() -> None:
    """The plotting helper draws one collection for each provided class."""
    namespace = load_function_namespace()
    linear_data = np.loadtxt(
        fname=DATA_DIRECTORY / "svm_dataset1.csv",
        delimiter=",",
        skiprows=1,
    )
    figure, axes = plt.subplots(figsize=(6, 4))

    returned_axes = namespace["plot_data"](
        features=linear_data[:, :2],
        labels=linear_data[:, 2],
        x_label="x1",
        y_label="x2",
        positive_label="Positive",
        negative_label="Negative",
        x_minimum=0.0,
        x_maximum=4.2,
        y_minimum=0.0,
        y_maximum=5.0,
        axes=axes,
    )

    assert returned_axes is axes
    assert len(axes.collections) == 2
    plt.close(fig=figure)
