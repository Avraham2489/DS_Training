"""Reusable helpers for the clustering and PCA recitation."""

import logging

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(name="clustering_pca_recitation")


def plot_clusters(
    *,
    clusters_points: list[NDArray[np.float64]],
    title: str,
    cluster_names: list[str] | None = None,
) -> Figure:
    """Plot each two-dimensional cluster in a different color.

    Args:
        clusters_points: Point matrices grouped by cluster.
        title: Figure title.
        cluster_names: Optional legend labels matching the cluster order.

    Returns:
        Matplotlib figure containing the cluster plot.

    Raises:
        ValueError: If a cluster is empty or the label count is inconsistent.
    """
    if cluster_names is not None and len(cluster_names) != len(clusters_points):
        raise ValueError("cluster_names must match the number of clusters.")

    figure, axes = plt.subplots(figsize=(7, 5))
    for cluster_number, current_cluster_points in enumerate(
        clusters_points,
        start=1,
    ):
        cluster_array = np.asarray(
            a=current_cluster_points,
            dtype=np.float64,
        )
        if cluster_array.size == 0:
            error_message = f"Cluster {cluster_number} contains no points."
            LOGGER.error(msg=error_message)
            raise ValueError(error_message)
        cluster_name = (
            cluster_names[cluster_number - 1]
            if cluster_names is not None
            else f"Cluster {cluster_number}"
        )
        axes.scatter(
            x=cluster_array[:, 0],
            y=cluster_array[:, 1],
            s=20,
            alpha=0.75,
            label=cluster_name,
        )
    axes.set(
        title=title,
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    axes.legend()
    return figure


def get_closest_center(
    *,
    point: NDArray[np.floating],
    centers: NDArray[np.floating],
) -> int:
    """Return the index of the Euclidean-nearest center.

    Args:
        point: One feature vector.
        centers: Matrix with one center per row.

    Returns:
        Index of the nearest center.
    """
    point_array = np.asarray(a=point, dtype=np.float64)
    center_array = np.asarray(a=centers, dtype=np.float64)
    center_distances = np.linalg.norm(
        x=center_array - point_array,
        axis=1,
    )
    return int(np.argmin(a=center_distances))


def kmeans(
    *,
    data: NDArray[np.floating],
    clusters_num: int,
    random_seed: int,
    eps: float = 1e-8,
    maximum_iterations: int = 300,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Cluster data by alternating assignment and center updates.

    Args:
        data: Feature matrix with one sample per row.
        clusters_num: Required number of clusters.
        random_seed: Seed used to select the first center.
        eps: Maximum center movement considered converged.
        maximum_iterations: Maximum number of update steps.

    Returns:
        Point matrices grouped by cluster and the final centers.

    Raises:
        ValueError: If an argument is invalid.
        RuntimeError: If a cluster becomes empty or convergence is not reached.
    """
    feature_data = np.asarray(a=data, dtype=np.float64)
    if feature_data.ndim != 2 or not np.isfinite(feature_data).all():
        raise ValueError("data must be a finite two-dimensional matrix.")
    if not 1 <= clusters_num <= feature_data.shape[0]:
        raise ValueError("clusters_num must be between 1 and the sample count.")
    if eps <= 0.0 or maximum_iterations <= 0:
        raise ValueError("eps and maximum_iterations must be positive.")

    random_generator = np.random.default_rng(seed=random_seed)
    center_indices = [
        int(
            random_generator.integers(
                low=0,
                high=feature_data.shape[0],
            )
        )
    ]
    while len(center_indices) < clusters_num:
        selected_centers = feature_data[center_indices]
        squared_distances = np.sum(
            a=(
                feature_data[:, np.newaxis, :]
                - selected_centers[np.newaxis, :, :]
            )
            ** 2,
            axis=2,
        )
        minimum_squared_distances = np.min(
            a=squared_distances,
            axis=1,
        )
        minimum_squared_distances[center_indices] = -np.inf
        center_indices.append(int(np.argmax(a=minimum_squared_distances)))
    centers = feature_data[center_indices].copy()

    for current_iteration in range(maximum_iterations):
        assignments = np.array(
            object=[
                get_closest_center(point=current_point, centers=centers)
                for current_point in feature_data
            ],
            dtype=np.int64,
        )
        updated_centers = np.empty_like(prototype=centers)
        for cluster_index in range(clusters_num):
            current_cluster_points = feature_data[assignments == cluster_index]
            if len(current_cluster_points) == 0:
                error_message = (
                    f"K-Means produced an empty cluster at iteration "
                    f"{current_iteration + 1}."
                )
                LOGGER.error(msg=error_message)
                raise RuntimeError(error_message)
            updated_centers[cluster_index] = current_cluster_points.mean(axis=0)

        maximum_center_shift = float(
            np.max(
                a=np.linalg.norm(
                    x=updated_centers - centers,
                    axis=1,
                )
            )
        )
        centers = updated_centers
        if maximum_center_shift <= eps:
            final_assignments = np.array(
                object=[
                    get_closest_center(point=current_point, centers=centers)
                    for current_point in feature_data
                ],
                dtype=np.int64,
            )
            cluster_points = [
                feature_data[final_assignments == cluster_index].copy()
                for cluster_index in range(clusters_num)
            ]
            LOGGER.info(
                msg=f"K-Means converged in {current_iteration + 1} iterations."
            )
            return cluster_points, centers

    error_message = (
        f"K-Means did not converge within {maximum_iterations} iterations."
    )
    LOGGER.error(msg=error_message)
    raise RuntimeError(error_message)


def get_neighbors(
    *,
    point_index: int,
    data: NDArray[np.floating],
    eps: float,
) -> list[int]:
    """Return indices whose distance from one point is at most eps.

    Args:
        point_index: Row index of the neighborhood center.
        data: Feature matrix with one sample per row.
        eps: Inclusive neighborhood radius.

    Returns:
        Neighbor indices, including point_index itself.

    Raises:
        ValueError: If eps or the data shape is invalid.
        IndexError: If point_index is outside the data.
    """
    feature_data = np.asarray(a=data, dtype=np.float64)
    if feature_data.ndim != 2 or not np.isfinite(feature_data).all():
        raise ValueError("data must be a finite two-dimensional matrix.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if not 0 <= point_index < feature_data.shape[0]:
        raise IndexError("point_index is outside the data matrix.")

    distances = np.linalg.norm(
        x=feature_data - feature_data[point_index],
        axis=1,
    )
    return np.flatnonzero(distances <= eps).astype(dtype=np.int64).tolist()


def dbscan(
    *,
    data: NDArray[np.floating],
    eps: float,
    min_points: int = 1,
) -> NDArray[np.int64]:
    """Cluster density-connected samples and label noise as -1.

    Args:
        data: Feature matrix with one sample per row.
        eps: Inclusive neighborhood radius.
        min_points: Minimum neighborhood size, including the point itself.

    Returns:
        Integer cluster labels, with -1 reserved for noise.

    Raises:
        ValueError: If an argument is invalid.
    """
    feature_data = np.asarray(a=data, dtype=np.float64)
    if feature_data.ndim != 2 or not np.isfinite(feature_data).all():
        raise ValueError("data must be a finite two-dimensional matrix.")
    if eps <= 0.0 or min_points <= 0:
        raise ValueError("eps and min_points must be positive.")

    unassigned_label = 0
    noise_label = -1
    labels = np.full(
        shape=feature_data.shape[0],
        fill_value=unassigned_label,
        dtype=np.int64,
    )
    visited = np.zeros(shape=feature_data.shape[0], dtype=bool)
    cluster_counter = 0

    for point_index in range(feature_data.shape[0]):
        if visited[point_index]:
            continue

        visited[point_index] = True
        neighbors = get_neighbors(
            point_index=point_index,
            data=feature_data,
            eps=eps,
        )
        if len(neighbors) < min_points:
            labels[point_index] = noise_label
            continue

        cluster_counter += 1
        labels[point_index] = cluster_counter
        seed_points = list(neighbors)
        seed_members = set(seed_points)
        current_seed_position = 0

        while current_seed_position < len(seed_points):
            neighbor_index = seed_points[current_seed_position]
            current_seed_position += 1

            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                expanded_neighbors = get_neighbors(
                    point_index=neighbor_index,
                    data=feature_data,
                    eps=eps,
                )
                if len(expanded_neighbors) >= min_points:
                    for expanded_neighbor_index in expanded_neighbors:
                        if expanded_neighbor_index not in seed_members:
                            seed_points.append(expanded_neighbor_index)
                            seed_members.add(expanded_neighbor_index)

            if labels[neighbor_index] <= unassigned_label:
                labels[neighbor_index] = cluster_counter

    return labels


def pca(
    *,
    data: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform centered data onto the right-singular-vector basis.

    Args:
        data: Feature matrix with one sample per row.

    Returns:
        PCA scores and components stored as rows.

    Raises:
        ValueError: If data is not a finite two-dimensional matrix.
    """
    feature_data = np.asarray(a=data, dtype=np.float64)
    if feature_data.ndim != 2 or not np.isfinite(feature_data).all():
        raise ValueError("data must be a finite two-dimensional matrix.")

    centered_data = feature_data - feature_data.mean(axis=0)
    _, _, right_singular_vectors = np.linalg.svd(
        a=centered_data,
        full_matrices=False,
    )
    components = right_singular_vectors
    transformed_data = centered_data @ components.T
    return transformed_data, components
