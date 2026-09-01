"""Reusable helpers for the unsupervised-learning exercise."""

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture


ClusteringCase = tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    dict[str, Any],
]
ClusteringCases = dict[str, ClusteringCase]
ClusteringAssignments = dict[str, dict[str, NDArray[np.int64]]]


def read_gray_image(*, path: str | Path) -> NDArray[np.uint8]:
    """Read one image as an unsigned 8-bit grayscale matrix.

    Args:
        path: Image file path.

    Returns:
        Grayscale image indexed by height and width.

    Raises:
        ValueError: If OpenCV cannot read the image.
    """
    color_image = cv2.imread(filename=str(path), flags=cv2.IMREAD_COLOR)
    if color_image is None:
        raise ValueError(f"OpenCV could not read {path}")
    return cv2.cvtColor(src=color_image, code=cv2.COLOR_BGR2GRAY)


def create_clustering_failure_cases(*, random_seed: int) -> ClusteringCases:
    """Create one deterministic failure case for each clustering method.

    Args:
        random_seed: Seed used for data generation and model initialization.

    Returns:
        Named feature matrices, true labels, and estimators.
    """
    random_generator = np.random.default_rng(seed=random_seed)

    large_cluster = random_generator.normal(
        loc=(0.0, 0.0),
        scale=0.8,
        size=(1_200, 2),
    )
    small_cluster = random_generator.normal(
        loc=(3.5, 0.0),
        scale=0.12,
        size=(60, 2),
    )
    kmeans_features = np.vstack(tup=(large_cluster, small_cluster))
    kmeans_labels = np.hstack(
        tup=(
            np.zeros(shape=len(large_cluster), dtype=np.int64),
            np.ones(shape=len(small_cluster), dtype=np.int64),
        )
    )

    dense_cluster = random_generator.normal(
        loc=(-2.0, 0.0),
        scale=0.12,
        size=(500, 2),
    )
    diffuse_cluster = random_generator.normal(
        loc=(2.0, 0.0),
        scale=0.65,
        size=(500, 2),
    )
    dbscan_features = np.vstack(tup=(dense_cluster, diffuse_cluster))
    dbscan_labels = np.hstack(
        tup=(
            np.zeros(shape=len(dense_cluster), dtype=np.int64),
            np.ones(shape=len(diffuse_cluster), dtype=np.int64),
        )
    )

    gmm_centers = np.array(
        object=[[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]],
        dtype=np.float64,
    )
    gmm_features = np.vstack(
        tup=tuple(
            random_generator.normal(
                loc=current_center,
                scale=0.35,
                size=(250, 2),
            )
            for current_center in gmm_centers
        )
    )
    gmm_labels = np.repeat(
        a=np.arange(start=0, stop=4, dtype=np.int64),
        repeats=250,
    )
    identical_precisions = np.repeat(
        a=np.eye(N=2)[None, :, :],
        repeats=4,
        axis=0,
    )

    return {
        "KMeans weakness: imbalanced clusters": (
            kmeans_features,
            kmeans_labels,
            {
                "KMeans": KMeans(
                    n_clusters=2,
                    n_init=10,
                    random_state=random_seed,
                ),
                "DBSCAN": DBSCAN(eps=0.60, min_samples=5, n_jobs=-1),
                "Gaussian mixture": GaussianMixture(
                    n_components=2,
                    n_init=10,
                    random_state=random_seed,
                ),
            },
        ),
        "DBSCAN weakness: unequal densities": (
            dbscan_features,
            dbscan_labels,
            {
                "KMeans": KMeans(
                    n_clusters=2,
                    n_init=10,
                    random_state=random_seed,
                ),
                "DBSCAN": DBSCAN(eps=0.12, min_samples=10, n_jobs=-1),
                "Gaussian mixture": GaussianMixture(
                    n_components=2,
                    n_init=10,
                    random_state=random_seed,
                ),
            },
        ),
        "GMM weakness: poor initialization": (
            gmm_features,
            gmm_labels,
            {
                "KMeans": KMeans(
                    n_clusters=4,
                    n_init=10,
                    random_state=random_seed,
                ),
                "DBSCAN": DBSCAN(eps=0.45, min_samples=10, n_jobs=-1),
                "Gaussian mixture": GaussianMixture(
                    n_components=4,
                    n_init=1,
                    means_init=np.zeros(shape=(4, 2)),
                    weights_init=np.full(shape=4, fill_value=0.25),
                    precisions_init=identical_precisions,
                    random_state=random_seed,
                ),
            },
        ),
    }


def evaluate_clustering_cases(
    *,
    clustering_cases: ClusteringCases,
) -> tuple[pd.DataFrame, ClusteringAssignments]:
    """Fit all estimators and measure clustering quality and runtime.

    Args:
        clustering_cases: Cases returned by ``create_clustering_failure_cases``.

    Returns:
        Result table and predicted assignments for visualization.
    """
    assignments: ClusteringAssignments = {}
    result_rows = []
    for case_name, (features, true_labels, estimators) in clustering_cases.items():
        assignments[case_name] = {"Ground truth": true_labels}
        for algorithm_name, estimator in estimators.items():
            fit_start_time = perf_counter()
            predicted_labels = estimator.fit_predict(X=features)
            elapsed_seconds = perf_counter() - fit_start_time
            assignments[case_name][algorithm_name] = predicted_labels
            result_rows.append(
                {
                    "dataset": case_name,
                    "algorithm": algorithm_name,
                    "ARI": adjusted_rand_score(
                        labels_true=true_labels,
                        labels_pred=predicted_labels,
                    ),
                    "seconds": elapsed_seconds,
                    "clusters_found": np.unique(ar=predicted_labels).size,
                }
            )

    results = pd.DataFrame(data=result_rows)
    results["ARI"] = results["ARI"].round(decimals=3)
    results["seconds"] = results["seconds"].round(decimals=4)
    return results, assignments


def validate_clustering_results(
    *,
    clustering_cases: ClusteringCases,
    clustering_results: pd.DataFrame,
    random_seed: int,
) -> float:
    """Validate the intended failures and the corrected GMM initialization.

    Args:
        clustering_cases: Evaluated clustering cases.
        clustering_results: Table returned by ``evaluate_clustering_cases``.
        random_seed: Seed for the corrected GMM.

    Returns:
        ARI achieved by the robustly initialized GMM.

    Raises:
        AssertionError: If a failure case does not behave as intended.
    """
    expected_failure_limits = {
        ("KMeans weakness: imbalanced clusters", "KMeans"): 0.50,
        ("DBSCAN weakness: unequal densities", "DBSCAN"): 0.85,
        ("GMM weakness: poor initialization", "Gaussian mixture"): 0.50,
    }
    for (
        case_name,
        algorithm_name,
    ), maximum_allowed_ari in expected_failure_limits.items():
        observed_ari = clustering_results.loc[
            (clustering_results["dataset"] == case_name)
            & (clustering_results["algorithm"] == algorithm_name),
            "ARI",
        ].iloc[0].item()
        if observed_ari >= maximum_allowed_ari:
            raise AssertionError(
                f"{algorithm_name} did not fail on {case_name}: "
                f"ARI={observed_ari:.3f}"
            )

    successful_pairs = (
        ("KMeans weakness: imbalanced clusters", "DBSCAN"),
        ("KMeans weakness: imbalanced clusters", "Gaussian mixture"),
        ("DBSCAN weakness: unequal densities", "KMeans"),
        ("DBSCAN weakness: unequal densities", "Gaussian mixture"),
        ("GMM weakness: poor initialization", "KMeans"),
        ("GMM weakness: poor initialization", "DBSCAN"),
    )
    for case_name, algorithm_name in successful_pairs:
        observed_ari = clustering_results.loc[
            (clustering_results["dataset"] == case_name)
            & (clustering_results["algorithm"] == algorithm_name),
            "ARI",
        ].iloc[0].item()
        if observed_ari < 0.95:
            raise AssertionError(
                f"{algorithm_name} did not recover {case_name}: "
                f"ARI={observed_ari:.3f}"
            )

    gmm_features, gmm_labels, _ = clustering_cases[
        "GMM weakness: poor initialization"
    ]
    robust_gmm = GaussianMixture(
        n_components=4,
        init_params="k-means++",
        n_init=10,
        random_state=random_seed,
    )
    robust_labels = robust_gmm.fit_predict(X=gmm_features)
    robust_ari = adjusted_rand_score(
        labels_true=gmm_labels,
        labels_pred=robust_labels,
    )
    if robust_ari < 0.95:
        raise AssertionError(
            f"Robust GMM initialization produced ARI={robust_ari:.3f}"
        )
    return float(robust_ari)


def plot_clustering_assignments(
    *,
    clustering_cases: ClusteringCases,
    clustering_assignments: ClusteringAssignments,
) -> Figure:
    """Plot truth and the three fitted assignments for every case.

    Args:
        clustering_cases: Evaluated clustering cases.
        clustering_assignments: Labels returned by ``evaluate_clustering_cases``.

    Returns:
        Matplotlib comparison figure.
    """
    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(14, 10),
        constrained_layout=True,
    )
    algorithm_names = ("Ground truth", "KMeans", "DBSCAN", "Gaussian mixture")
    for row_position, (case_name, (features, _, _)) in enumerate(
        clustering_cases.items()
    ):
        for column_position, algorithm_name in enumerate(algorithm_names):
            axes[row_position, column_position].scatter(
                x=features[:, 0],
                y=features[:, 1],
                c=clustering_assignments[case_name][algorithm_name],
                cmap="tab10",
                s=7,
                alpha=0.75,
            )
            axes[row_position, column_position].set_title(
                label=f"{case_name}\n{algorithm_name}",
                fontsize=9,
            )
    return figure


def reconstruct_with_pca(
    *,
    pca_model: PCA,
    samples: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Reconstruct samples through a fitted PCA model.

    Args:
        pca_model: Fitted PCA model.
        samples: Two-dimensional sample matrix.

    Returns:
        Reconstructed samples with the same shape.
    """
    low_dimensional_samples = pca_model.transform(X=samples)
    return pca_model.inverse_transform(X=low_dimensional_samples)


def fit_class_pca_models(
    *,
    training_samples: NDArray[np.floating],
    training_labels: NDArray[np.integer],
    class_count: int,
    component_count: int,
    random_seed: int,
    iterated_power: int | str = "auto",
) -> list[PCA]:
    """Fit one PCA model to each labeled class subset.

    Args:
        training_samples: Flattened training images.
        training_labels: Integer class label for every training image.
        class_count: Number of class-specific models to fit.
        component_count: Components retained by each model.
        random_seed: Seed for randomized SVD.
        iterated_power: Power iterations used by randomized SVD.

    Returns:
        Fitted PCA models ordered by integer class label.
    """
    class_models = []
    for class_index in range(class_count):
        class_samples = training_samples[training_labels == class_index]
        class_model = PCA(
            n_components=component_count,
            svd_solver="randomized",
            iterated_power=iterated_power,
            random_state=random_seed,
        )
        class_model.fit(X=class_samples)
        class_models.append(class_model)
    return class_models


def predict_by_reconstruction_error(
    *,
    pca_models: list[PCA],
    samples: NDArray[np.floating],
) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
    """Choose the PCA model with the smallest per-sample reconstruction MSE.

    Args:
        pca_models: One fitted PCA model per class.
        samples: Samples to classify.

    Returns:
        Predicted model indices and the complete reconstruction-error matrix.
    """
    reconstruction_error_columns = []
    for current_pca_model in pca_models:
        reconstructed_samples = reconstruct_with_pca(
            pca_model=current_pca_model,
            samples=samples,
        )
        reconstruction_error_columns.append(
            ((samples - reconstructed_samples) ** 2).mean(axis=1)
        )
    reconstruction_errors = np.stack(
        arrays=reconstruction_error_columns,
        axis=1,
    )
    predictions = reconstruction_errors.argmin(axis=1).astype(dtype=np.int64)
    return predictions, reconstruction_errors


def calculate_component_prefix_errors(
    *,
    pca_model: PCA,
    samples: NDArray[np.floating],
    component_counts: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Calculate reconstruction MSE for prefixes of one fitted PCA basis.

    Args:
        pca_model: PCA fitted with at least the largest requested component count.
        samples: Samples to reconstruct.
        component_counts: Increasing numbers of leading components to retain.

    Returns:
        Matrix with one sample per row and one component count per column.
    """
    centered_samples = samples - pca_model.mean_
    component_scores = pca_model.transform(X=samples)
    total_squared_error = (centered_samples**2).sum(axis=1)
    return np.stack(
        arrays=[
            (
                total_squared_error
                - (component_scores[:, :current_component_count] ** 2).sum(axis=1)
            )
            / samples.shape[1]
            for current_component_count in component_counts
        ],
        axis=1,
    )


def evaluate_pca_component_counts(
    *,
    training_samples: NDArray[np.floating],
    training_labels: NDArray[np.integer],
    test_samples: NDArray[np.floating],
    test_labels: NDArray[np.integer],
    component_counts: NDArray[np.integer],
    class_count: int,
    random_seed: int,
) -> tuple[list[PCA], pd.DataFrame]:
    """Evaluate class-specific PCA using successive component prefixes.

    Args:
        training_samples: Flattened training images.
        training_labels: Integer labels for the training images.
        test_samples: Flattened images to classify.
        test_labels: Integer labels for the test images.
        component_counts: Component counts to evaluate.
        class_count: Number of class-specific PCA models.
        random_seed: Seed for randomized SVD.

    Returns:
        Maximum-component PCA models and the accuracy table.
    """
    maximum_component_count = component_counts.max().item()
    maximum_component_models = fit_class_pca_models(
        training_samples=training_samples,
        training_labels=training_labels,
        class_count=class_count,
        component_count=maximum_component_count,
        random_seed=random_seed,
        iterated_power=2,
    )
    error_curves_by_model = [
        calculate_component_prefix_errors(
            pca_model=current_pca_model,
            samples=test_samples,
            component_counts=component_counts,
        )
        for current_pca_model in maximum_component_models
    ]

    accuracies = []
    for component_position in range(len(component_counts)):
        current_error_matrix = np.stack(
            arrays=[
                model_error_curves[:, component_position]
                for model_error_curves in error_curves_by_model
            ],
            axis=1,
        )
        predictions = current_error_matrix.argmin(axis=1)
        accuracies.append(
            accuracy_score(y_true=test_labels, y_pred=predictions)
        )

    results = pd.DataFrame(
        data={
            "n_components": component_counts,
            "accuracy": accuracies,
        }
    )
    return maximum_component_models, results
