"""Tests for the completed unsupervised-learning exercise notebook."""

import importlib.util
from pathlib import Path

import cv2
import nbformat
import numpy as np
from sklearn.decomposition import PCA


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "D - ML"
    / "unsupervised"
    / "exercise"
    / "__clustering_exercise.ipynb"
)
IMAGE_DIRECTORY = NOTEBOOK_PATH.parent.parent / "images"
UTILITY_PATH = NOTEBOOK_PATH.parent / "unsupervised_utils.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    name="unsupervised_utils",
    location=UTILITY_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Unable to load unsupervised utilities from {UTILITY_PATH}")
UNSUPERVISED_UTILS = importlib.util.module_from_spec(spec=MODULE_SPEC)
MODULE_SPEC.loader.exec_module(module=UNSUPERVISED_UTILS)


def load_resized_grayscale_images(
    *,
    celebrity_name: str,
    image_count: int,
) -> np.ndarray:
    """Load a deterministic real-image subset for a compact unit test.

    Args:
        celebrity_name: Directory name under the exercise image directory.
        image_count: Number of numerically sorted JPEG images to load.

    Returns:
        Flattened grayscale images with one image per row.
    """
    image_paths = sorted(
        (IMAGE_DIRECTORY / celebrity_name).glob(pattern="*.jpg"),
        key=lambda image_path: int(image_path.stem),
    )[:image_count]
    if len(image_paths) != image_count:
        raise FileNotFoundError(
            f"Expected {image_count} images for {celebrity_name}; "
            f"found {len(image_paths)}"
        )

    flattened_images = []
    for image_path in image_paths:
        grayscale_image = UNSUPERVISED_UTILS.read_gray_image(
            path=image_path,
        )
        resized_image = cv2.resize(
            src=grayscale_image,
            dsize=(32, 32),
            interpolation=cv2.INTER_AREA,
        )
        flattened_images.append(resized_image.reshape(-1))
    return np.stack(arrays=flattened_images, axis=0).astype(dtype=np.float32)


def test_pca_helpers_reconstruct_and_classify_real_images() -> None:
    """Class-specific PCA reconstructs its own real training images best."""
    alicia_images = load_resized_grayscale_images(
        celebrity_name="Alicia Vikander",
        image_count=8,
    )
    andy_images = load_resized_grayscale_images(
        celebrity_name="Andy Serkis",
        image_count=8,
    )
    pca_models = [
        PCA(n_components=7, svd_solver="full").fit(X=alicia_images),
        PCA(n_components=7, svd_solver="full").fit(X=andy_images),
    ]
    samples = np.stack(
        arrays=[alicia_images[0], andy_images[0]],
        axis=0,
    )

    reconstructed_samples = UNSUPERVISED_UTILS.reconstruct_with_pca(
        pca_model=pca_models[0],
        samples=samples,
    )
    predictions, reconstruction_errors = (
        UNSUPERVISED_UTILS.predict_by_reconstruction_error(
            pca_models=pca_models,
            samples=samples,
        )
    )

    assert reconstructed_samples.shape == samples.shape
    assert np.isfinite(reconstructed_samples).all()
    assert reconstruction_errors.shape == (2, 2)
    assert np.isfinite(reconstruction_errors).all()
    np.testing.assert_array_equal(
        actual=predictions,
        desired=np.array(object=[0, 1]),
    )


def test_clustering_cases_have_one_intended_failure_each() -> None:
    """Every synthetic case isolates the weakness named in the notebook."""
    clustering_cases = UNSUPERVISED_UTILS.create_clustering_failure_cases(
        random_seed=42
    )
    clustering_results, _ = UNSUPERVISED_UTILS.evaluate_clustering_cases(
        clustering_cases=clustering_cases
    )
    robust_gmm_ari = UNSUPERVISED_UTILS.validate_clustering_results(
        clustering_cases=clustering_cases,
        clustering_results=clustering_results,
        random_seed=42,
    )

    assert robust_gmm_ari >= 0.95


def test_notebook_keeps_prompts_and_is_fully_executed() -> None:
    """The exercise prompts remain present and every code cell ran cleanly."""
    notebook = nbformat.read(fp=NOTEBOOK_PATH, as_version=4)
    cell_sources = [cell.source for cell in notebook.cells]
    combined_source = "\n".join(cell.source for cell in notebook.cells)
    original_fragments = (
        "# Unsupervised Learning - Exercise",
        "### your code here",
        "img = cv2.imread('images/Alicia Vikander/1.jpg')[:,:,::-1]",
        "First we load the face images",
        "celebs = ['Alicia Vikander', 'Amy Adams', 'Andy Serkis']",
        "Perform the prediction process with PCA models",
    )
    for original_fragment in original_fragments:
        assert original_fragment in combined_source

    required_additions = (
        "clustering_failure_cases.png",
        "from unsupervised_utils import reconstruct_with_pca",
        "predict_by_reconstruction_error",
        "celebrity_confusion_matrix",
        "maximum_component_pca_models",
        "Timestamped output files",
    )
    for required_addition in required_additions:
        assert required_addition in combined_source

    assert not any(
        line.lstrip().startswith("def ")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
    )
    sequencing_fragments = (
        "First we load the face images",
        "from unsupervised_utils import read_gray_image",
        "Now make a PCA model with n_components=10",
        "predict_by_reconstruction_error",
        "Perform the prediction process with PCA models",
        "evaluate_pca_component_counts",
    )
    sequencing_positions = {
        fragment: next(
            cell_position
            for cell_position, cell_source in enumerate(cell_sources)
            if fragment in cell_source
        )
        for fragment in sequencing_fragments
    }
    assert sequencing_positions["First we load the face images"] < (
        sequencing_positions["from unsupervised_utils import read_gray_image"]
    )
    assert sequencing_positions["Now make a PCA model with n_components=10"] < (
        sequencing_positions["predict_by_reconstruction_error"]
    )
    assert sequencing_positions[
        "Perform the prediction process with PCA models"
    ] < sequencing_positions["evaluate_pca_component_counts"]

    nonempty_code_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip()
    ]
    assert all(cell.execution_count is not None for cell in nonempty_code_cells)
    error_outputs = [
        output
        for cell in nonempty_code_cells
        for output in cell.outputs
        if output.output_type == "error"
    ]
    assert not error_outputs


def test_notebook_created_all_expected_plot_files() -> None:
    """The latest timestamped execution contains every requested plot."""
    output_root = NOTEBOOK_PATH.parent / "outputs"
    output_directories = sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert output_directories
    latest_output_directory = output_directories[0]
    expected_plot_names = (
        "clustering_failure_cases.png",
        "color_quantization.png",
        "color_inertia_curve.png",
        "image_shapes.png",
        "pca_components.png",
        "pca_reconstruction.png",
        "pca_confusion_matrix.png",
        "pca_component_accuracy.png",
    )
    for plot_name in expected_plot_names:
        plot_path = latest_output_directory / plot_name
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0

    notebook = nbformat.read(fp=NOTEBOOK_PATH, as_version=4)
    stream_text = "\n".join(
        output.text
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream"
    )
    for expected_log_fragment in (
        "Robustly initialized GMM ARI: 1.000",
        "10-component PCA classifier accuracy:",
        "Best bonus accuracy:",
        "Timestamped output files:",
    ):
        assert expected_log_fragment in stream_text
