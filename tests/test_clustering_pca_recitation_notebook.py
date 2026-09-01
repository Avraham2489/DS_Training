"""Tests for the clustering and PCA recitation notebook and utilities."""

import importlib.util
from pathlib import Path

import nbformat
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "D - ML"
    / "unsupervised"
    / "recitation"
    / "Clustering and PCA - recitation .ipynb"
)
RANDOM_SEED = 42
UTILITY_PATH = NOTEBOOK_PATH.parent / "unsupervised_utils.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    name="recitation_unsupervised_utils",
    location=UTILITY_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Unable to load recitation utilities from {UTILITY_PATH}")
UNSUPERVISED_UTILS = importlib.util.module_from_spec(spec=MODULE_SPEC)
MODULE_SPEC.loader.exec_module(module=UNSUPERVISED_UTILS)

BLOB_CENTERS = np.array(
    object=[[-4.0, -2.0], [0.0, 4.0], [4.0, -1.0]],
    dtype=np.float64,
)
BLOB_STANDARD_DEVIATIONS = [0.55, 0.75, 0.65]


def create_recitation_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create the deterministic dataset used by the notebook.

    Returns:
        Two-dimensional samples and their generating cluster labels.
    """
    feature_data, true_labels = make_blobs(
        n_samples=500,
        centers=BLOB_CENTERS,
        cluster_std=BLOB_STANDARD_DEVIATIONS,
        random_state=RANDOM_SEED,
    )
    return feature_data.astype(dtype=np.float64), true_labels.astype(dtype=np.int64)


def test_kmeans_is_deterministic_and_recovers_the_blobs() -> None:
    """K-Means assigns every sample and recovers the generating groups."""
    feature_data, true_labels = create_recitation_data()

    cluster_points, centers = UNSUPERVISED_UTILS.kmeans(
        data=feature_data,
        clusters_num=3,
        eps=1e-8,
        random_seed=RANDOM_SEED,
        maximum_iterations=300,
    )
    repeated_cluster_points, repeated_centers = UNSUPERVISED_UTILS.kmeans(
        data=feature_data,
        clusters_num=3,
        eps=1e-8,
        random_seed=RANDOM_SEED,
        maximum_iterations=300,
    )
    predicted_labels = np.array(
        object=[
            UNSUPERVISED_UTILS.get_closest_center(point=point, centers=centers)
            for point in feature_data
        ],
        dtype=np.int64,
    )

    assert centers.shape == (3, 2)
    assert sum(len(current_cluster) for current_cluster in cluster_points) == len(
        feature_data
    )
    assert all(len(current_cluster) > 0 for current_cluster in cluster_points)
    assert adjusted_rand_score(
        labels_true=true_labels,
        labels_pred=predicted_labels,
    ) >= 0.99
    np.testing.assert_allclose(actual=centers, desired=repeated_centers)
    for current_cluster, repeated_cluster in zip(
        cluster_points,
        repeated_cluster_points,
        strict=True,
    ):
        np.testing.assert_allclose(actual=current_cluster, desired=repeated_cluster)


def test_dbscan_matches_sklearn_on_the_recitation_data() -> None:
    """The custom density expansion agrees with scikit-learn DBSCAN."""
    feature_data, true_labels = create_recitation_data()

    custom_labels = UNSUPERVISED_UTILS.dbscan(
        data=feature_data,
        eps=0.55,
        min_points=5,
    )
    sklearn_labels = DBSCAN(eps=0.55, min_samples=5).fit_predict(X=feature_data)

    assert custom_labels.dtype == np.int64
    assert adjusted_rand_score(
        labels_true=sklearn_labels,
        labels_pred=custom_labels,
    ) == 1.0
    assert adjusted_rand_score(
        labels_true=true_labels,
        labels_pred=custom_labels,
    ) >= 0.95


def test_pca_matches_sklearn_and_does_not_mutate_input() -> None:
    """Full PCA reconstructs the input and matches sklearn up to component signs."""
    feature_data, _ = create_recitation_data()
    original_feature_data = feature_data.copy()

    transformed_data, components = UNSUPERVISED_UTILS.pca(data=feature_data)
    reconstructed_data = transformed_data @ components + feature_data.mean(axis=0)
    sklearn_pca = PCA(n_components=2, svd_solver="full").fit(X=feature_data)
    component_alignment = np.abs(
        np.diag(v=components @ sklearn_pca.components_.T)
    )

    np.testing.assert_array_equal(actual=feature_data, desired=original_feature_data)
    np.testing.assert_allclose(
        actual=transformed_data.mean(axis=0),
        desired=np.zeros(shape=2),
        atol=1e-12,
    )
    np.testing.assert_allclose(actual=reconstructed_data, desired=feature_data, atol=1e-12)
    np.testing.assert_allclose(
        actual=components @ components.T,
        desired=np.eye(N=2),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual=component_alignment,
        desired=np.ones(shape=2),
        atol=1e-12,
    )


def test_notebook_preserves_prompts_and_runs_in_order() -> None:
    """Original prompts remain and all nonempty code cells ran without errors."""
    notebook = nbformat.read(fp=NOTEBOOK_PATH, as_version=4)
    cell_sources = [cell.source for cell in notebook.cells]
    combined_source = "\n".join(cell_sources)
    original_fragments = (
        "# Clustering an PCA - recitation",
        "Start by generatig data and visualise it",
        "Implement the KMeans algorithm",
        "Implement the DBSCAN algorithm",
        "Don't forget to first change the mean of your data to 0",
        "Compare with sklearn.decomposition.PCA",
    )
    for original_fragment in original_fragments:
        assert original_fragment in combined_source

    sequencing_fragments = (
        "Implement the KMeans algorithm",
        "from unsupervised_utils import get_closest_center, kmeans",
        "Implement the DBSCAN algorithm",
        "from unsupervised_utils import dbscan, get_neighbors",
        "Don't forget to first change the mean of your data to 0",
        "from unsupervised_utils import pca",
    )
    sequencing_positions = {
        fragment: next(
            cell_position
            for cell_position, cell_source in enumerate(cell_sources)
            if fragment in cell_source
        )
        for fragment in sequencing_fragments
    }
    assert sequencing_positions["Implement the KMeans algorithm"] < (
        sequencing_positions[
            "from unsupervised_utils import get_closest_center, kmeans"
        ]
    )
    assert sequencing_positions["Implement the DBSCAN algorithm"] < (
        sequencing_positions["from unsupervised_utils import dbscan, get_neighbors"]
    )
    assert sequencing_positions[
        "Don't forget to first change the mean of your data to 0"
    ] < sequencing_positions["from unsupervised_utils import pca"]

    assert "from unsupervised_init import LOGGER, RANDOM_SEED" in combined_source
    assert not any(
        line.lstrip().startswith("def ")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
    )

    nonempty_code_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip()
    ]
    assert all(cell.execution_count is not None for cell in nonempty_code_cells)
    assert not any(
        output.output_type == "error"
        for cell in nonempty_code_cells
        for output in cell.outputs
    )
